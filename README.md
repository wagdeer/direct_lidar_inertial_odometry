# Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction

#### [[ IEEE ICRA ](https://ieeexplore.ieee.org/document/10160508)] [[ arXiv ](https://arxiv.org/abs/2203.03749)] [[ Video ](https://www.youtube.com/watch?v=4-oXjG8ow10)] [[ Presentation ](https://www.youtube.com/watch?v=Hmiw66KZ1tU)]

DLIO is a new lightweight LiDAR-inertial odometry algorithm with a novel coarse-to-fine approach in constructing continuous-time trajectories for precise motion correction. It features several algorithmic improvements over its predecessor, [DLO](https://github.com/vectr-ucla/direct_lidar_odometry), and was presented at the IEEE International Conference on Robotics and Automation (ICRA) in London, UK in 2023.

<br>
<p align='center'>
    <img src="./doc/img/dlio.png" alt="drawing" width="720"/>
</p>

## Instructions

### Sensor Setup
DLIO has been extensively tested using a variety of sensor configurations and currently supports Livox LiDARs. The point cloud should be of input type `sensor_msgs::PointCloud2` and the 6-axis IMU input type of `sensor_msgs::Imu`.

For best performance, extrinsic calibration between the LiDAR/IMU sensors and the robot's center-of-gravity should be inputted into `cfg/dlio.yaml`. If the exact values of these are unavailable, a rough LiDAR-to-IMU extrinsics can also be used (note however that performance will be degraded).

IMU intrinsics are also necessary for best performance, and there are several open-source calibration tools to get these values. These values should also go into `cfg/dlio.yaml`. In practice however, if you are just testing this work, using the default ideal values and performing the initial calibration procedure should be fine.

Also note that the LiDAR and IMU sensors _need_ to be properly time-synchronized, otherwise DLIO will not work. We recommend using a LiDAR with an integrated IMU for simplicity of extrinsics and synchronization.

### Dependencies
The following has been verified to be compatible, although other configurations may work too:

- Ubuntu 22.04
- ROS Humble (`rclcpp`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `pcl_ros`)
- C++ 17
- CMake >= `3.12.4`
- OpenMP >= `4.5`
- Point Cloud Library >= `1.10.0`
- Eigen >= `3.3.7`
- [Equivariant Preintegration](https://github.com/aau-cns/equivariant-preintegration) — Lie-group IMU preintegration
- [Lie++](https://github.com/aau-cns/Lie-plusplus) — header-only Lie group library (dependency of Equivariant Preintegration)

```sh
sudo apt install libomp-dev libpcl-dev libeigen3-dev
```

The Equivariant Preintegration and Lie++ libraries are header-only and must be cloned alongside the DLIO source:

```sh
cd ~/ros2_ws/src
git clone https://github.com/vectr-ucla/direct_lidar_inertial_odometry -b feature/ros2
git clone https://github.com/aau-cns/equivariant-preintegration.git
git clone https://github.com/aau-cns/Lie-plusplus.git
```

DLIO currently supports `ROS 1` and `ROS 2`!

### Compiling
Compile using the [`catkin_tools`](https://catkin-tools.readthedocs.io/en/latest/) package via:

```sh
mkdir ~/ros2_ws && cd ~/ros2_ws && mkdir src && cd src
```
```sh
git clone https://github.com/vectr-ucla/direct_lidar_inertial_odometry -b feature/ros2
```
```sh
cd ~/ros2_ws
```
```sh
colcon build --symlink-install --packages-select direct_lidar_inertial_odometry
```

### Execution

<details>
<summary> After compiling, don't forget to source before ROS commands.</summary>

``` bash
source ~/ros2_ws/install/setup.bash
```
</details>

Execute via:

```sh
roslaunch direct_lidar_inertial_odometry dlio.launch \
  rviz:={true, false} \
  pointcloud_topic:=/robot/lidar \
  imu_topic:=/robot/imu
```

<details>
<summary> Example command: </summary>

``` bash
ros2 launch direct_lidar_inertial_odometry dlio.launch.py rviz:=true pointcloud_topic:=/lexus3/os_center/points imu_topic:=/lexus3/os_center/imu
```
</details>

Be sure to change the topic names to your corresponding topics. Alternatively, edit the launch file directly if desired. If successful, you should see the following output in your terminal:
<br>
<p align='center'>
    <img src="./doc/img/terminal.png" alt="drawing" width="480"/>
</p>

### Services
To save DLIO's generated map into `.pcd` format, call the following service:

```sh
ros2 service call /save_pcd direct_lidar_inertial_odometry/srv/SavePCD "{'leaf_size': 0.2, 'save_path': '~/map'}"
```

### Test Data
For your convenience, we provide test data [here](https://drive.google.com/file/d/1Sp_Mph4rekXKY2euxYxv6SD6WIzB-wVU/view?usp=sharing) (1.2GB, 1m 13s, Ouster OS1-32) of an aggressive motion to test our motion correction scheme, and [here](https://drive.google.com/file/d/1HbmF5gTHxCAMqBkEd5PTxDNQvcI8tKXn/view?usp=sharing) (16.5GB, 4m 21s, Ouster OSDome) of a longer trajectory outside with lots of trees. Try these two datasets with both deskewing on and off!

<br>
<p align='center'>
    <img src="./doc/gif/aggressive.gif" alt="drawing" width="720"/>
</p>

## Citation
If you found this work useful, please cite our manuscript:

```bibtex
@article{chen2022dlio,
  title={Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction},
  author={Chen, Kenny and Nemiroff, Ryan and Lopez, Brett T},
  journal={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={3983-3989},
  doi={10.1109/ICRA48891.2023.10160508}
}
```

## Acknowledgements

We thank the authors of the [FastGICP](https://github.com/SMRT-AIST/fast_gicp) and [NanoFLANN](https://github.com/jlblancoc/nanoflann) open-source packages:

- Kenji Koide, Masashi Yokozuka, Shuji Oishi, and Atsuhiko Banno, “Voxelized GICP for Fast and Accurate 3D Point Cloud Registration,” in _IEEE International Conference on Robotics and Automation (ICRA)_, IEEE, 2021, pp. 11 054–11 059.
- Jose Luis Blanco and Pranjal Kumar Rai, “NanoFLANN: a C++ Header-Only Fork of FLANN, A Library for Nearest Neighbor (NN) with KD-Trees,” https://github.com/jlblancoc/nanoflann, 2014.

We would also like to thank Helene Levy and David Thorne for their help with data collection.

## Improvements (this fork)

This fork includes significant performance and correctness improvements over the upstream `feature/ros2` branch:

### IMU Integration
- Replaced manual first-order quaternion integration with **[Equivariant Preintegration](https://arxiv.org/abs/2411.05548)** (RA-L 2025), a Lie-group-based method offering better numerical stability, built-in covariance propagation, and SIMD-accelerated Eigen operations.

### Thread & CPU Optimization
- **OpenMP thread explosion fix** — Disabled nested parallelism in PCL (`omp_set_nested(0)`, `omp_set_max_active_levels(1)`) and capped OMP threads via launch file environment variables. Eliminated ~1000 idle OMP threads on multi-core systems (1049 → 89 threads, -91%).
- **Thread pool sizing** — Replaced hardcoded `thread_pool(4)` with `hardware_concurrency`-aware dynamic allocation.
- **Deskew coarse-grained parallelism** — Replaced per-timestamp `submit_loop` with block-based partitioning (4096-point minimum block), reducing pool task count from ~1000 to ~16 for Livox scans.
- **Async metrics with atomic gate** — Snapshot-based metrics computation with `exchange(true)` gate prevents stacking duplicate tasks. Thread-safe LPF state via `std::optional`.
- **Batched keyframe publish** — Merged N × `detach_task` per keyframe into a single batched publish task. Also fixes a race on `kf_pose_ros.poses`.
- **Convex/concave hull caching** — Skip recomputation when keyframe set and alpha parameters are unchanged.

### Code Quality
- Removed 4 unused `std::thread` members (dead code).
- Replaced `static` local variables in IMU calibration with thread-safe member variables.
- Fixed dead branch in `setAdaptiveParams` where density was unconditionally overwritten.
- Replaced `std::priority_queue` in `pushSubmapIndices` with `std::nth_element` (O(n log k) → O(n)).

### Performance Summary
| Metric | Before | After |
|--------|--------|-------|
| Threads (idle) | 1049 | ~57 |
| Threads (bag playback) | 1049 | ~89 |
| RSS (bag playback) | ~150 MB | ~90 MB |

## License
This work is licensed under the terms of the MIT license.

<br>
<p align='center'>
    <img src="./doc/img/ucla.png" alt="drawing" width="720"/>
</p>
<p align='center'>
    <img src="./doc/img/trees.png" alt="drawing" width="720"/>
</p>
