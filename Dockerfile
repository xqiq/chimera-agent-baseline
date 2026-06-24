# =============================================================================
# Grand Challenge container — slim build.
#
# Builds on python:3.12-slim and pip-installs a pinned vLLM instead of the
# ~30 GB vllm/vllm-openai base image. Same vLLM 0.23.0 / CUDA 13 wheels, only
# the packages we actually use — the image is roughly half the size.
#
# Runtime notes:
#   * Gemma 4's heterogeneous attention heads force vLLM's Triton attention
#     backend, which JIT-compiles kernels at startup — hence gcc/g++ (Triton
#     bundles its own ptxas, so no CUDA devel toolkit is needed).
#   * torch.compile/CUDA graphs are disabled via generation.enforce_eager, and
#     the flashinfer JIT sampler via VLLM_USE_FLASHINFER_SAMPLER=0, so nothing
#     else needs to compile against nvcc at runtime.
#   * Caches point at /tmp (the only guaranteed-writable dir on the platform)
#     and the HF hub is forced offline (no network at runtime).
#
# Input:   /input          (read-only, mounted by platform)
# Output:  /output         (writable, agent writes results here)
# Model:   /opt/ml/model   (read-only, weights uploaded separately)
# App:     /opt/app        (read-only, application code + resources)
# =============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    VLLM_CACHE_ROOT=/tmp/vllm \
    TRITON_CACHE_DIR=/tmp/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    VLLM_USE_FLASHINFER_SAMPLER=0

# gcc/g++: Triton runtime JIT compilation. libgomp1/libnuma1: OpenMP for
# scikit-learn/scipy and NUMA for vLLM.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by Grand Challenge)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# Install the pinned dependency closure (vLLM + torch + our deps). Fully
# deterministic — every build installs the exact same versions.
COPY --chown=user:user requirements.lock pyproject.toml README.md /opt/app/
RUN pip install --no-cache-dir -r requirements.lock

# Copy application code and install the package itself (deps already satisfied
# by the lockfile above, so --no-deps).
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user inference.py /opt/app/
RUN pip install --no-cache-dir --no-deps .

# Configs (single source of truth — same file Hydra reads locally), Jinja
# prompt templates, and runtime resources (RAG DB + embedding model).
COPY --chown=user:user configs/ /opt/app/configs/
COPY --chown=user:user templates/ /opt/app/templates/
COPY --chown=user:user resources/ /opt/app/resources/

USER user

ENTRYPOINT ["python3", "inference.py"]
