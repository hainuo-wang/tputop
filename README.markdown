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

![TPUTOP interface](/screenshot/NVTOP_ex1.png)

Quick Start
-----------

### Running TPUTOP

```bash
# Run tputop (auto-discovers TPU Pod workers)
./build/src/tputop

# Or with manual worker list
export NVTOP_TPU_POD_FILE=~/podips.txt
./build/src/tputop
```

### Interactive Commands

| Key | Action |
|-----|--------|
| F2 | Setup window |
| F6 | Sort processes |
| F9 | Kill process |
| F10 / q | Quit |
| F12 | Save config |

TPU Pod Support
---------------

TPUTOP automatically discovers all workers in a TPU Pod using the GCP metadata service. It reads the `worker-network-endpoints` attribute to find all worker IPs.

### Auto-Discovery

On GCP TPU VMs, TPUTOP automatically:
1. Queries metadata service for worker endpoints
2. Filters out the local worker
3. Connects to remote workers via SSH to collect TPU metrics

### Manual Configuration

If auto-discovery doesn't work, create a file with worker IPs:

```bash
# Create worker list file
cat > ~/podips.txt << EOF
10.130.0.25
10.130.0.24
10.130.0.22
EOF

# Set environment variable
export NVTOP_TPU_POD_FILE=~/podips.txt
```

Requirements
------------

- Google Cloud TPU VM
- `libtpuinfo.so` library (included with TPU runtime)
- SSH access to other workers in the Pod (passwordless)
- Python 3 on all workers

Build
-----

```bash
# Install dependencies
sudo apt install cmake libncurses5-dev libncursesw5-dev

# Build
mkdir -p build && cd build
cmake .. -DTPU_SUPPORT=ON
make -j$(nproc)

# Run
./src/tputop
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

Troubleshooting
---------------

### Remote workers not showing

1. Check SSH connectivity:
```bash
ssh -o BatchMode=yes <worker-ip> 'echo ok'
```

2. Check libtpuinfo on remote worker:
```bash
ssh <worker-ip> 'python3 -c "import ctypes; lib=ctypes.CDLL(\"libtpuinfo.so\"); print(lib.tpu_chip_count())"'
```

### Slow startup

The initial connection to remote workers may take a few seconds. After startup, data is cached and refreshed at 20fps.

License
-------

TPUTOP is based on NVTOP and is licensed under GPLv3.

Original NVTOP: https://github.com/Syllo/nvtop
