import shutil


def _is_first_task_on_local_node(task_ip_list, partition_id):
    task_ip = task_ip_list[partition_id]
    return task_ip not in task_ip_list[:partition_id]


def _get_local_task_num(task_ip_list, partition_id):
    num = 0
    task_ip = task_ip_list[partition_id]
    for ip in task_ip_list:
        if ip == task_ip:
            num += 1
    return num


def _create_ray_tmp_dir():
    from pyspark.files import SparkFiles
    import tempfile

    return tempfile.mkdtemp(prefix="Ray-tmp-")


def get_safe_port(ip):
    import socket
    """Returns an ephemeral port that is very likely to be free to bind to."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((ip, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _check_port_open(host, port):
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


def _get_ray_run_mapper(ray_app_fn, sparkContext):
    """
    `ray_app_fn` is a ray application function without `ray.init` part,
    and return a result object which is picklable.
    """

    _spark_task_cpus = int(sparkContext.getConf().get("spark.task.cpus", "1"))
    _spark_task_gpus = int(sparkContext.getConf().get("spark.task.resource.gpu.amount"))

    def mapper(x):
        from pyspark.taskcontext import BarrierTaskContext
        from mlflow.utils.process import _exec_cmd
        import time
        import ray
        import cloudpickle
        import sys
        import os

        context = BarrierTaskContext.get()
        context.barrier()

        partition_id = context.partitionId()
        task_ip_list = [info.address.split(":")[0] for info in context.getTaskInfos()]

        cuda_visible_devices_env_list = context.allGather(os.environ['CUDA_VISIBLE_DEVICES'])
        task_ip = task_ip_list[partition_id]

        gpu_id_str_list = []
        for i, cuda_visible_devices_env in enumerate(cuda_visible_devices_env_list):
            if task_ip_list[i] == task_ip:
                for gpu_id_str in cuda_visible_devices_env.split(','):
                    gpu_id_str_list.append(gpu_id_str.strip())

        ray_node_cuda_visible_devices_env = ','.join(gpu_id_str_list)
        ray_node_extra_envs = {'CUDA_VISIBLE_DEVICES': ray_node_cuda_visible_devices_env}

        head_ip = task_ip_list[0]

        is_ray_head_node = (partition_id == 0)
        is_first_local_task = _is_first_task_on_local_node(task_ip_list, partition_id)
        if is_ray_head_node:
            # Generate ray head node port in head task.
            ray_head_node_port_str = str(get_safe_port(task_ip))
        else:
            ray_head_node_port_str = ""

        # use BarrierTaskContext.allGather to broadcast head node port to all tasks
        ray_head_node_port = int(context.allGather(ray_head_node_port_str)[0])

        ray_node_proc = None
        ray_tmp_dir = None

        ray_exec_path = os.path.join(os.path.dirname(sys.executable), "ray")
        try:
            if is_first_local_task:
                # Only run Ray node on first local spark task.
                num_local_tasks = _get_local_task_num(task_ip_list, partition_id)
                num_cpus_for_ray_node = num_local_tasks * _spark_task_cpus
                num_gpus_for_ray_node = num_local_tasks * _spark_task_gpus
                ray_tmp_dir = _create_ray_tmp_dir()

                # start ray node.
                # TODO: set memory limitation
                ray_cmd = [
                    ray_exec_path, "start", f"--temp-dir={ray_tmp_dir}",
                    f"--num-cpus={num_cpus_for_ray_node}",
                    f"--num-gpus={num_gpus_for_ray_node}",
                    "--block"
                ]
                if is_ray_head_node:
                    ray_cmd.append("--head")
                    ray_cmd.append(f"--port={ray_head_node_port}")
                    ray_cmd.append("--include-dashboard=false")
                else:
                    ray_cmd.append(f"--address={head_ip}:{ray_head_node_port}")

            # Start Ray head node
            # TODO: Retry if starting failed (might because of port conflicts)
            # TODO: capture Ray node output and if error happen raise error with tail logs.
            if is_ray_head_node:
                ray_node_proc = _exec_cmd(
                    ray_cmd, synchronous=False, capture_output=False, stream_output=False,
                    extra_env=ray_node_extra_envs
                )

            for _ in range(40):
                time.sleep(1)
                if _check_port_open(head_ip, ray_head_node_port):
                    break

            if not _check_port_open(head_ip, ray_head_node_port):
                raise RuntimeError("Start Ray head node failed!")

            # Make all tasks wait for the Ray head node ready.
            context.barrier()

            # Start Ray worker node
            # TODO: Retry if starting failed (might because of port conflicts)
            if not is_ray_head_node and is_first_local_task:
                ray_node_proc = _exec_cmd(
                    ray_cmd, synchronous=False, capture_output=False, stream_output=False,
                    extra_env=ray_node_extra_envs
                )

            if is_ray_head_node:
                time.sleep(5)  # wait ray worker node
                # Run Ray application on head task
                ray.init(address=f"{head_ip}:{ray_head_node_port}")
                print("-------Ray-cluster-resources-------:\n" + str(ray.cluster_resources()) + "\n")

                result = ray_app_fn()
                yield cloudpickle.dumps(result)

            # The barrier here is used for ensuring all worker tasks keeping running until head task
            # ray application completes.
            context.barrier()
        finally:
            if ray_tmp_dir is not None:
                shutil.rmtree(ray_tmp_dir, ignore_errors=True)
            # TODO: set up PR_SET_PDEATHSIG signal, so that when spark job is killed,
            #  the child ray node process will be killed.
            if ray_node_proc is not None:
                ray_node_proc.terminate()

    return mapper


class RayRunner:

    def __init__(self, num_spark_tasks):
        self.num_spark_tasks = num_spark_tasks

    def run(self, ray_app_fn):
        from pyspark.sql import SparkSession
        import cloudpickle

        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext

        mapper = _get_ray_run_mapper(ray_app_fn, sc)
        result = sc.parallelize(range(self.num_spark_tasks), self.num_spark_tasks) \
            .barrier() \
            .mapPartitions(mapper).collect()[0]

        return cloudpickle.loads(result)
