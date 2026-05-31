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

```sh
sudo apt install libomp-dev libpcl-dev libeigen3-dev
```

Equivariant IMU preintegration and the Lie++ group library are **vendored** in `include/dlio/preintegration/` and `include/dlio/lie/` — no external cloning required.

### Compiling

```sh
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/wagdeer/direct_lidar_inertial_odometry -b feature/humble
cd ~/ros2_ws
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

This fork (`feature/humble`) includes significant improvements over upstream DLIO:

### IMU Integration
- Replaced manual first-order quaternion integration with **[Equivariant Preintegration](https://arxiv.org/abs/2411.05548)** (RA-L 2025) — Lie-group-based method on the Gal(3) manifold offering better numerical stability and built-in covariance propagation.
- Unified `propagateState()` (geometric observer) with the same Lie-group integrator, eliminating Euler integration drift and reducing correction workload on the observer. This is the primary contributor to improved Z-axis stability.

### Thread & CPU Optimization
- **OpenMP thread explosion fix** — Capped OMP threads and disabled nested parallelism, eliminating ~1000 idle threads on multi-core systems (1049 → 89).
- **Deskew coarse-grained parallelism** — Block-based partitioning (≥4096 pts/block) reduces pool task count from ~1000 to ~16 for Livox scans.
- **Async metrics** — Snapshot + atomic gate pattern moves spaciousness/density computation off the critical path.
- **Hull caching** — Convex/concave hull recomputation skipped when keyframe set unchanged.
- **Submap index O(n) selection** — `std::nth_element` replaces full sort for top-K keyframe selection.

### Robustness
- **T_corr race fix** — Snapshot GICP correction before async publish, preventing stale transforms.
- **numProcessors fallback** — `sysconf(_SC_NPROCESSORS_ONLN)` when `/proc/cpuinfo` unavailable (containers).
- **`fopen` null guard** — Prevents crash when `/proc/cpuinfo` is inaccessible.
- **Static locals → members** — `transformImu()` no longer shares state across instances.

### Code Quality
- **Zero external deps** — Equivariant Preintegration and Lie++ vendored into `include/dlio/` (BSD-2-Clause, with attribution).
- **Debug gate** — `debug: false` (default) disables all statistics collection and terminal dashboard for zero runtime overhead. Set `debug: true` for real-time CPU/memory monitoring with circular-buffer-capped (200-entry) statistics.
- **Removed C++14 override** — `-std=c++14` compile flag was overriding `CXX_STANDARD 17`.
- **Parameterized IMU rate** — `odom/imu/nominalRate` replaces hardcoded 200 Hz fallback.
- **Removed 10+ unused member variables** — Dead code cleanup from upstream ROS1 migration.

### Platform Support
- Tested on **x86-64** (AMD Ryzen 9950X, Ubuntu 22.04 + ROS Humble + Docker) and **ARM** (NVIDIA Jetson Orin NX 16G).
- For Jetson: add `-march=armv8.2-a+fp16+dotprod` to CMakeLists for NEON acceleration. Reduce voxel resolution (`0.5`) and GICP iterations (`16`) if needed.

## License
This work is licensed under the terms of the MIT license.

The vendored libraries in `include/dlio/preintegration/` and `include/dlio/lie/` are
licensed under BSD-2-Clause with a non-commercial condition. Copyright © University of
Klagenfurt — Control of Networked Systems (CNS). See individual file headers for details.

<br>
<p align='center'>
    <img src="./doc/img/ucla.png" alt="drawing" width="720"/>
</p>
<p align='center'>
    <img src="./doc/img/trees.png" alt="drawing" width="720"/>
</p>
