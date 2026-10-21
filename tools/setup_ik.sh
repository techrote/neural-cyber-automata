#!/usr/bin/env bash
# Isolated optional IK runtime. No device workload is executed by setup.
set -euo pipefail

lane="${1:-cuda}"
if [[ $# -gt 1 || ( "$lane" != "cuda" && "$lane" != "cpu" ) ]]; then
    echo "Usage: bash tools/setup_ik.sh [cuda|cpu]" >&2
    exit 2
fi

ik_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ik_environment="$ik_project_root/.venv-ik"
if [[ -e "$ik_environment" || -L "$ik_environment" ]]; then
    echo "Refusing to modify existing .venv-ik. Use a fresh checkout for another runtime identity." >&2
    exit 2
fi
if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Python 3.12 with venv support is required." >&2
    exit 2
fi

python3.12 -m venv "$ik_environment"
ik_python="$ik_environment/bin/python"
"$ik_python" -m pip install setuptools==84.0.0 wheel==0.45.1 numpy==2.3.5
if [[ "$lane" == "cuda" ]]; then
    ik_torch_version="2.8.0+cu126"
    ik_wheel_index="https://download.pytorch.org/whl/cu126"
else
    ik_torch_version="2.8.0+cpu"
    ik_wheel_index="https://download.pytorch.org/whl/cpu"
fi
"$ik_python" -m pip install "torch==$ik_torch_version" --index-url "$ik_wheel_index"
"$ik_python" -m pip install --no-build-isolation -e "$ik_project_root"
"$ik_python" -c 'import sys; from importlib.metadata import version; assert sys.version_info[:2] == (3, 12); assert version("numpy") == "2.3.5"; assert version("torch") == sys.argv[1]; print("Optional IK package versions verified; device qualification has not run.")' "$ik_torch_version"

echo "Setup complete. Activate with: source .venv-ik/bin/activate"
echo "Read docs/IK_RUNTIME.md before declaring or executing a bounded smoke run."
