TPUTOP
======

What is TPUTOP?
---------------

TPUTOP is a TPU monitoring tool based on NVTOP, providing htop-like task monitoring for Google Cloud TPU Pods. It supports monitoring multiple TPU devices across all workers in a TPU Pod, displaying real-time utilization, memory usage, and process information.

**Key Features:**
- Monitor all TPU devices in a TPU Pod (local + remote workers)
- Auto-discovery of TPU Pod workers via GCP metadata service
- Real-time TPU utilization and memory monitoring
- Process information display (PID, USER, CPU, Memory, Command)
- 20fps refresh rate for real-time monitoring

![TPUTOP interface](/screenshot/tputop.png)

Quick Start
-----------

### Running TPUTOP

```bash
tputop
```

TPU Pod Support
---------------

TPUTOP automatically discovers all workers in a TPU Pod using the GCP metadata service. It reads the `worker-network-endpoints` attribute to find all worker IPs, filters out the local worker, and connects to remote workers via SSH to collect TPU metrics.

Requirements
------------

- Google Cloud TPU VM
- `libtpuinfo.so` library (included with TPU runtime)
- SSH access to other workers in the Pod (passwordless)
- Python 3 on all workers

Installation
------------

### Main Node (Full Installation)

```bash
# Install build dependencies
sudo apt install -y libdrm-dev libsystemd-dev libudev-dev cmake libncurses5-dev libncursesw5-dev git

# Install libtpuinfo
wget https://github.com/rdyro/libtpuinfo/releases/download/v0.0.1/libtpuinfo-linux-x86_64.so
sudo mv libtpuinfo-linux-x86_64.so /lib/libtpuinfo.so

# Clone and build
git clone https://github.com/hainuo-wang/tputop.git
cd tputop && mkdir build && cd build
cmake -DTPU_SUPPORT=ON ..
make

# Install
sudo make install
```

### Other Worker Nodes (libtpuinfo Only)

For TPU pods with multiple workers, other nodes only need libtpuinfo installed.
The main node will connect via SSH to collect TPU metrics.

```bash
# Install libtpuinfo only (no need to install tputop)
wget https://github.com/rdyro/libtpuinfo/releases/download/v0.0.1/libtpuinfo-linux-x86_64.so
sudo mv libtpuinfo-linux-x86_64.so /lib/libtpuinfo.so
```

Displayed Information
---------------------

### Device Panel

For each TPU device:
- **Device Name**: TPU model and device ID (e.g., "TPU v4 [0@10.130.0.25]")
- **TPU Utilization**: Duty cycle percentage
- **Memory Usage**: Used / Total HBM memory

### Process Panel

| Column | Description |
|--------|-------------|
| PID | Process ID |
| USER | Process owner |
| DEV | TPU device index |
| TYPE | Process type (Compute) |
| TPU | TPU utilization % |
| TPU MEM | TPU memory usage |
| CPU | CPU usage % |
| HOST MEM | Host memory usage |
| Command | Process command line |


License
-------

TPUTOP is based on NVTOP and is licensed under GPLv3.

Acknowledgements
----------------

- [NVTOP](https://github.com/Syllo/nvtop) - The original GPU monitoring tool this project is based on
- [Google TPU Research Cloud (TRC)](https://sites.research.google/trc/) - For providing TPU resources for development and testing
