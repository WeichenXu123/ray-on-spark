# Ray on spark
[Ray on spark](https://github.com/WeichenXu123/ray-on-spark)

This repository provides a ray spark plugin which supports running ray on spark cluster.

## Basic example

```
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
```
