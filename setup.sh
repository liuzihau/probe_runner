#!/usr/bin/env bash
# probe_runner/setup.sh — fetch external dependencies for the probe experiment.
#
# Run this from the directory that contains the probe_runner/ folder (NOT from inside it).
# It clones Fast-dLLM into ./external/Fast-dLLM/ at a pinned commit, then installs Python deps.
#
# After this script completes, you can run:
#   python -m probe_runner.run_probes --model llada
#
# Override the install dir by passing FAST_DLLM_V1_PATH=/your/path before running.
# In that case the script will skip the clone and only verify the path.

set -euo pipefail

FAST_DLLM_REPO="https://github.com/NVlabs/Fast-dLLM.git"
# Pin to a known-good commit. Update if you need a newer version.
FAST_DLLM_REF="${FAST_DLLM_REF:-main}"

if [[ -n "${FAST_DLLM_V1_PATH:-}" ]]; then
    echo "FAST_DLLM_V1_PATH is set to: ${FAST_DLLM_V1_PATH}"
    if [[ ! -f "${FAST_DLLM_V1_PATH}/llada/model/modeling_llada.py" ]]; then
        echo "ERROR: ${FAST_DLLM_V1_PATH}/llada/model/modeling_llada.py not found." >&2
        echo "Either fix FAST_DLLM_V1_PATH or unset it to let setup.sh clone Fast-dLLM here." >&2
        exit 1
    fi
    echo "Fast-dLLM v1 already present at FAST_DLLM_V1_PATH — skipping clone."
else
    DEFAULT_PARENT="external"
    DEFAULT_DIR="${DEFAULT_PARENT}/Fast-dLLM"
    mkdir -p "${DEFAULT_PARENT}"
    if [[ -d "${DEFAULT_DIR}/.git" ]]; then
        echo "Fast-dLLM already cloned at ${DEFAULT_DIR}; pulling latest of ${FAST_DLLM_REF}."
        git -C "${DEFAULT_DIR}" fetch origin "${FAST_DLLM_REF}"
        git -C "${DEFAULT_DIR}" checkout "${FAST_DLLM_REF}"
        git -C "${DEFAULT_DIR}" pull --ff-only origin "${FAST_DLLM_REF}" || true
    else
        echo "Cloning Fast-dLLM into ${DEFAULT_DIR} ..."
        git clone "${FAST_DLLM_REPO}" "${DEFAULT_DIR}"
        git -C "${DEFAULT_DIR}" checkout "${FAST_DLLM_REF}"
    fi
    if [[ ! -f "${DEFAULT_DIR}/v1/llada/model/modeling_llada.py" ]]; then
        echo "ERROR: clone seemed to succeed but ${DEFAULT_DIR}/v1/llada/model/modeling_llada.py is missing." >&2
        exit 1
    fi
    echo "Fast-dLLM v1 ready at ${DEFAULT_DIR}/v1"
fi

echo
echo "Installing Python dependencies from probe_runner/requirements.txt ..."
python -m pip install -r probe_runner/requirements.txt

echo
echo "Setup complete. Try a smoke test:"
echo "    python -m probe_runner.run_probes --model llada --n_samples 2"
