#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-0}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
VLLM_PACKAGE="${VLLM_PACKAGE:-vllm>=0.12.0}"

if [[ -d "${WORKSPACE_DIR}" ]]; then
  CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_DIR}/.cache}"
  OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-${WORKSPACE_DIR}/ollama-models}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_DIR}}"
else
  CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache}"
  OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-${HOME}/.ollama/models}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/data}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export OLLAMA_MODELS="${OLLAMA_MODELS_DIR}"
export PATH="${HOME}/.local/bin:${PATH}"

GREEN=$'\033[1;32m'
CYAN=$'\033[0;36m'
YELLOW=$'\033[0;33m'
RESET=$'\033[0m'

info() {
  printf '%s%s%s\n' "${CYAN}" "$1" "${RESET}"
}

success() {
  printf '%s%s%s\n' "${GREEN}" "$1" "${RESET}"
}

warn() {
  printf '%s%s%s\n' "${YELLOW}" "$1" "${RESET}"
}

append_env_once() {
  local line="$1"
  local shell_file="${HOME}/.bashrc"
  touch "${shell_file}"
  if ! grep -Fqx "${line}" "${shell_file}"; then
    printf '%s\n' "${line}" >> "${shell_file}"
  fi
}

install_system_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    info "Installing RunPod system packages"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      ca-certificates \
      curl \
      git \
      jq \
      less \
      pciutils \
      vim \
      zstd
    apt-get clean
    return
  fi
  warn "apt-get not found; skipping system package installation"
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    success "uv already installed: $(uv --version)"
    return
  fi
  info "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  success "uv installed: $(uv --version)"
}

install_ollama() {
  if [[ "${INSTALL_OLLAMA}" != "1" ]]; then
    warn "Skipping Ollama install because INSTALL_OLLAMA=${INSTALL_OLLAMA}"
    return
  fi
  if command -v ollama >/dev/null 2>&1; then
    success "Ollama already installed: $(ollama --version 2>/dev/null || true)"
    return
  fi
  info "Installing Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
  success "Ollama installed"
}

write_shell_env() {
  info "Persisting cache/model environment in ~/.bashrc"
  append_env_once "export PATH=\"${HOME}/.local/bin:\$PATH\""
  append_env_once "export UV_CACHE_DIR=\"${UV_CACHE_DIR}\""
  append_env_once "export HF_HOME=\"${HF_HOME}\""
  append_env_once "export TORCH_HOME=\"${TORCH_HOME}\""
  append_env_once "export PIP_CACHE_DIR=\"${PIP_CACHE_DIR}\""
  append_env_once "export OLLAMA_MODELS=\"${OLLAMA_MODELS}\""
}

sync_repo() {
  info "Running uv sync"
  cd "${REPO_DIR}"
  uv sync --python "${PYTHON_VERSION}"
}

install_vllm() {
  if [[ "${INSTALL_VLLM}" != "1" ]]; then
    warn "Skipping vLLM install because INSTALL_VLLM=${INSTALL_VLLM}"
    return
  fi
  info "Installing vLLM into the repo environment"
  cd "${REPO_DIR}"
  uv pip install "${VLLM_PACKAGE}"
}

print_next_steps() {
  success "RunPod setup complete"
  printf '\n'
  printf 'Environment:\n'
  printf '  OLLAMA_MODELS=%s\n' "${OLLAMA_MODELS}"
  printf '  UV_CACHE_DIR=%s\n' "${UV_CACHE_DIR}"
  printf '  HF_HOME=%s\n' "${HF_HOME}"
  printf '  TORCH_HOME=%s\n' "${TORCH_HOME}"
  printf '\n'
  printf 'Start vLLM in one terminal:\n'
  printf '  cd %s\n' "${REPO_DIR}"
  printf '  bash scripts/start_vllm_gemma4.sh\n'
  printf '\n'
  printf 'Then, in another terminal:\n'
  printf '  cd %s\n' "${REPO_DIR}"
  printf '  uv run synthetic-data-gen build --out %s/synthetic-10k --train-size 8000 --eval-size 2000 --generation-harness deepagent --generator-backend vllm --generator-model google/gemma-4-E4B-it --vllm-base-url http://localhost:8000/v1 --embedding-model BAAI/bge-small-en-v1.5 --embedding-device cuda --wandb-project finance-router-data-gen\n' "${OUTPUT_ROOT}"
}

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${OLLAMA_MODELS}"
install_system_packages
install_uv
install_ollama
write_shell_env
sync_repo
install_vllm
print_next_steps
