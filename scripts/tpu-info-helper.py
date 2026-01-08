#!/usr/bin/env python3
"""
Helper script to output TPU information in a simple format for nvtop remote monitoring.
Output format: count|device_id,memory_usage,total_memory,duty_cycle,pid|...
"""

import ctypes
import sys

def main():
    try:
        lib = ctypes.CDLL('libtpuinfo.so')
    except OSError:
        print("0")
        sys.exit(0)

    # Define function signatures
    lib.tpu_chip_count.restype = ctypes.c_int
    lib.tpu_pids.argtypes = [ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]
    lib.tpu_pids.restype = ctypes.c_int
    lib.tpu_metrics.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    lib.tpu_metrics.restype = ctypes.c_int

    count = lib.tpu_chip_count()
    if count <= 0:
        print("0")
        sys.exit(0)

    # Allocate arrays
    pids = (ctypes.c_longlong * count)()
    device_ids = (ctypes.c_longlong * count)()
    memory_usage = (ctypes.c_longlong * count)()
    total_memory = (ctypes.c_longlong * count)()
    duty_cycle = (ctypes.c_double * count)()

    # Get PIDs
    if lib.tpu_pids(pids, count) != 0:
        for i in range(count):
            pids[i] = -1

    # Get metrics (port -1 means default)
    lib.tpu_metrics(-1, device_ids, memory_usage, total_memory, duty_cycle, count)

    # Output format: count|dev_id,mem_used,mem_total,duty,pid|...
    parts = [str(count)]
    for i in range(count):
        parts.append(f"{device_ids[i]},{memory_usage[i]},{total_memory[i]},{duty_cycle[i]:.2f},{pids[i]}")

    print("|".join(parts))

if __name__ == "__main__":
    main()
