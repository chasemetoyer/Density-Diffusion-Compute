# Environment for every smoke-test command on a pod. Source it; don't run it.
# Caches, the virtual environment, uv's Python and the CUDA 13.0 toolkit live on the density volume (/workspace).
export PROJECT=/workspace/Density_Project
export UV_PROJECT_ENVIRONMENT=/workspace/venv
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/python
export UV_PYTHON_PREFERENCE=only-managed
export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_TELEMETRY=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0  # no second copy of the weights in the Xet chunk cache
export CUDA_HOME=/workspace/cuda-13.0
export TORCH_CUDA_ARCH_LIST="9.0;12.0"  # H100 (sm_90) and the RTX PRO 6000 Blackwell slice (sm_120)
export FORCE_CUDA=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS; must be set before cuBLAS initializes
export SMOKE_SCRATCH=/workspace/smoke_scratch
export PYTHONUNBUFFERED=1
# CPU thread pools default to every host core (128 on the slice's host) while the pod's cgroup allows a few
# CPUs (about 3.4 on the slice), so 64 threads thrash under throttling. Size them to the cgroup's CPU quota rounded
# down, so the pools never exceed it (Chase, 2026-10-07; density/cpu_threads.py sizes torch's and onnxruntime's).
cpu_quota=$(awk '$1 != "max" {q = int($1 / $2); print (q < 1 ? 1 : q)}' /sys/fs/cgroup/cpu.max 2>/dev/null)
export OMP_NUM_THREADS=${cpu_quota:-$(getconf _NPROCESSORS_ONLN)}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PATH="/workspace/bin:$UV_PROJECT_ENVIRONMENT/bin:$CUDA_HOME/bin:$PATH"
