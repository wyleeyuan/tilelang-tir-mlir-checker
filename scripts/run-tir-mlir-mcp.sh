#!/usr/bin/env bash
# Place beside the deployed tir_mlir_checker/ package on SG2044.
set -eo pipefail
{
    here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    sg_root="$HOME/sg2044-tilelang-riscv"
    source "$sg_root/checker/env.sh"
    unset LD_PRELOAD
    python="$sg_root/.venv/bin/python"
    mkdir -p -- "$here/tmp" "$here/runs"
    export TMPDIR="$here/tmp" PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
    export TIR_MLIR_DEFAULT_TIR_BACKEND=tvm
    export TIR_MLIR_DEFAULT_MLIR_BACKEND=official
    export TIR_MLIR_IR_PYTHON="$python"
    export TIR_MLIR_MLIR_OPT="$TILELANG_RISCV_LLVM_ROOT/bin/mlir-opt"
    export TIR_MLIR_DEFAULT_OUT_ROOT="$here/runs"
    cd -- "$here"
} >&2
exec "$python" -m tir_mlir_checker.mcp_server
