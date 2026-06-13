# SOP: Compiling llama.cpp-tq3 on Remote GPU Server (192.168.1.77)

## Overview

This document describes the standard operating procedure for compiling llama.cpp with TQ3_4S support and CUDA-enabled RPC on the remote GPU server at `192.168.1.77`.

**Remote Machine Specs**:
- OS: Ubuntu 22.04 (GLIBC 2.35)
- GPU: NVIDIA RTX 3090 (24 GB VRAM)
- CUDA: 12.9 (primary), 11.5 (legacy, do not use)
- GCC: 11.4
- Network: 1 Gbps Ethernet

**Build Location**: `~/code/llama.cpp-tq3/build-rpc-cuda12/`

---

## Prerequisites

### 1. SSH Access

```bash
ssh awee@192.168.1.77
# Password: (ask admin if you don't have it)
```

### 2. Verify CUDA 12.9 Installation

```bash
/usr/local/cuda-12.9/bin/nvcc --version
# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Tue_Oct_29_23:00:44_PDT_2024
# Cuda compilation tools, release 12.9, V12.9.86
```

**⚠️ Do NOT use CUDA 11.5** — it has C++17 compilation bugs that will cause build failures.

### 3. Verify GPU Availability

```bash
nvidia-smi
# Expected: RTX 3090 with ~24 GB free VRAM
```

---

## Step-by-Step Compilation

### Step 1: Navigate to Build Directory

```bash
cd ~/code/llama.cpp-tq3/build-rpc-cuda12
```

If the directory doesn't exist, create it:

```bash
cd ~/code/llama.cpp-tq3
mkdir -p build-rpc-cuda12
cd build-rpc-cuda12
```

### Step 2: Configure CMake

```bash
cmake .. \
  -DGGML_RPC=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

**Explanation of flags**:
- `-DGGML_RPC=ON`: Enable RPC support (required for distributed inference)
- `-DGGML_CUDA=ON`: Enable CUDA support (required for GPU acceleration)
- `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc`: Explicitly use CUDA 12.9 (avoids CUDA 11.5 bugs)
- `-DCMAKE_CUDA_ARCHITECTURES=86`: Target RTX 3090 (Ampere, compute 8.6) only — reduces build time from 45-60 min to 10-15 min

**Expected output**:
```
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found CUDAToolkit: /usr/local/cuda-12.9/targets/x86_64-linux/include (found version "12.9.86")
-- Enabling CUDA architectures: 86
-- Configuring done (2.3s)
-- Generating done (0.8s)
-- Build files have been written to: /home/awee/code/llama.cpp-tq3/build-rpc-cuda12
```

### Step 3: Build RPC Server and Libraries

```bash
cmake --build . --target ggml-rpc rpc-server -- -j12
```

**Expected output**:
```
[ 12%] Building C object ggml/src/CMakeFiles/ggml-rpc.dir/ggml-rpc.c.o
[ 25%] Building CXX object src/CMakeFiles/rpc-server.dir/rpc-server.cpp.o
[ 37%] Linking CXX shared library libggml-rpc.so
[ 50%] Linking CXX executable rpc-server
[100%] Built target ggml-rpc
[100%] Built target rpc-server
```

**Build time**: ~2-3 minutes (with `-j12`)

### Step 4: Verify Build Artifacts

```bash
ls -lh bin/rpc-server bin/libggml-rpc.so*
```

**Expected output**:
```
-rwxrwxr-x 1 awee awee  191K Jun  9 22:30 bin/rpc-server
lrwxrwxrwx 1 awee awee   18 Jun  9 22:30 bin/libggml-rpc.so -> libggml-rpc.so.0.13.1
lrwxrwxrwx 1 awee awee   18 Jun  9 22:30 bin/libggml-rpc.so.0 -> libggml-rpc.so.0.13.1
-rwxrwxr-x 1 awee awee  160K Jun  9 22:30 bin/libggml-rpc.so.0.13.1
```

**⚠️ Critical**: The library size must match the local machine's library size. If they differ, RPC will fail with `recv failed` errors.

### Step 5: Verify Library Size Match (Optional but Recommended)

On the **local machine**:

```bash
ls -l /home/awee/code/tan_llama/build/bin/libggml-rpc.so.0.13.1
# Example output: -rwxrwxr-x 1 awee awee 163272 Jun  9 22:30 libggml-rpc.so.0.13.1
```

On the **remote machine**:

```bash
ls -l ~/code/llama.cpp-tq3/build-rpc-cuda12/bin/libggml-rpc.so.0.13.1
# Should be: 163272 bytes (same as local)
```

If sizes differ, rebuild on remote:

```bash
cd ~/code/llama.cpp-tq3/build-rpc-cuda12
rm -f bin/libggml-rpc.so*
cmake --build . --target ggml-rpc -- -j12
```

---

## Starting the RPC Server

### Manual Start (for Testing)

```bash
cd ~/code/llama.cpp-tq3/build-rpc-cuda12
./bin/rpc-server --host 0.0.0.0 --port 50052
```

**Expected output**:
```
Starting RPC server on 0.0.0.0:50052
  Backend: CUDA
  Devices: 1
  Device 0: NVIDIA GeForce RTX 3090, VRAM: 24576 MB, Free: 23700 MB
Listening for connections...
```

### Background Start (for Production)

```bash
cd ~/code/llama.cpp-tq3/build-rpc-cuda12
nohup ./bin/rpc-server --host 0.0.0.0 --port 50052 > ~/rpc-server.log 2>&1 &
echo $! > ~/rpc-server.pid
```

**Verify it's running**:

```bash
ps aux | grep rpc-server
# Expected: ./bin/rpc-server --host 0.0.0.0 --port 50052

tail -f ~/rpc-server.log
# Expected: Listening for connections...
```

### Stop the RPC Server

```bash
kill $(cat ~/rpc-server.pid)
rm ~/rpc-server.pid
```

---

## Git Synchronization

### Critical Requirement

**Both local and remote machines MUST be on the same git commit** for RPC to work. If commits differ, you'll get `HELLO request size mismatch` errors.

### Check Current Commit

On **local machine**:

```bash
cd /home/awee/code/tan_llama
git rev-parse HEAD
# Example: 8ad718007a87e244a2a616e3a13a0b5b16e5d1e7
```

On **remote machine**:

```bash
cd ~/code/llama.cpp-tq3
git rev-parse HEAD
# Must match local: 8ad718007a87e244a2a616e3a13a0b5b16e5d1e7
```

### Sync to Specific Commit

If commits differ, sync remote to match local:

```bash
cd ~/code/llama.cpp-tq3
git fetch origin
git checkout 8ad718007a87e244a2a616e3a13a0b5b16e5d1e7
# Or use the commit hash from local machine
```

Then **rebuild** (see Step 3 above).

---

## Troubleshooting

### Problem: CUDA 11.5 Compilation Errors

**Symptom**:
```
/usr/local/cuda-11.5/bin/nvcc: error: unrecognized command line option '-std=c++17'
```

**Solution**: Explicitly specify CUDA 12.9:

```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
```

### Problem: RPC Connection Refused

**Symptom**:
```
Error: Failed to connect to RPC server at 192.168.1.77:50052
```

**Solution**:
1. Check if rpc-server is running: `ps aux | grep rpc-server`
2. Check firewall: `sudo ufw status` (should allow port 50052)
3. Check network: `ping 192.168.1.77`

### Problem: Library Size Mismatch

**Symptom**:
```
recv failed (bytes_recv=0, size_to_recv=8)
Remote RPC server crashed or returned malformed response
```

**Solution**: Rebuild RPC library on remote to match local:

```bash
cd ~/code/llama.cpp-tq3/build-rpc-cuda12
rm -f bin/libggml-rpc.so*
cmake --build . --target ggml-rpc -- -j12
```

### Problem: Git Commit Mismatch

**Symptom**:
```
HELLO request size mismatch (0 vs 24)
```

**Solution**: Sync git commits (see "Git Synchronization" section above).

### Problem: Out of Memory on Remote GPU

**Symptom**:
```
CUDA error: out of memory
```

**Solution**:
1. Check VRAM usage: `nvidia-smi`
2. Kill other GPU processes: `fuser -v /dev/nvidia*`
3. Reduce model size or context length

---

## Maintenance

### Weekly: Check for Updates

```bash
cd ~/code/llama.cpp-tq3
git fetch origin
git log HEAD..origin/main --oneline
# If there are new commits, coordinate with local machine to update
```

### Monthly: Clean Build Artifacts

```bash
cd ~/code/llama.cpp-tq3/build-rpc-cuda12
make clean
# Then rebuild (see Step 3)
```

### Quarterly: Update CUDA (if needed)

```bash
# Download new CUDA toolkit from NVIDIA
# Install to /usr/local/cuda-XX.X
# Update cmake command to use new version
# Rebuild
```

---

## Quick Reference

### Common Commands

```bash
# SSH to remote
ssh awee@192.168.1.77

# Navigate to build dir
cd ~/code/llama.cpp-tq3/build-rpc-cuda12

# Configure (if starting fresh)
cmake .. -DGGML_RPC=ON -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86

# Build
cmake --build . --target ggml-rpc rpc-server -- -j12

# Start RPC server
./bin/rpc-server --host 0.0.0.0 --port 50052

# Check status
ps aux | grep rpc-server
nvidia-smi

# Stop RPC server
pkill -f rpc-server
```

### File Locations

```
~/code/llama.cpp-tq3/              # Source code
~/code/llama.cpp-tq3/build-rpc-cuda12/  # Build directory
~/code/llama.cpp-tq3/build-rpc-cuda12/bin/rpc-server  # RPC server binary
~/code/llama.cpp-tq3/build-rpc-cuda12/bin/libggml-rpc.so.0.13.1  # RPC library
~/rpc-server.log                   # RPC server log (if started in background)
~/rpc-server.pid                   # RPC server PID (if started in background)
```

### Network Information

```
Remote IP: 192.168.1.77
RPC Port: 50052
Protocol: TCP
Bandwidth: 1 Gbps (125 MB/s)
Latency: ~0.5-1ms round trip
```

---

## Contact

If you encounter issues not covered in this document:
1. Check the troubleshooting section above
2. Review RPC server logs: `tail -f ~/rpc-server.log`
3. Contact the admin for hardware issues
4. Check llama.cpp GitHub issues for upstream bugs

---

**Last Updated**: 2026-06-09
**Maintained By**: Awee
**Git Commit**: 8ad718007
