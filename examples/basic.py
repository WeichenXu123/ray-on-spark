from rayonspark import RayRunner
import ray
import time

rayrunner = RayRunner(2)


def ray_app_fn():
    time.sleep(20)

    @ray.remote
    def f(x):
        import os
        import time
        time.sleep(20)
        return x * x

    futures = [f.remote(i) for i in range(2)]
    return ray.get(futures)


results = rayrunner.run(ray_app_fn)
print(results)
