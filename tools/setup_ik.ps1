# Isolated optional IK runtime. No device workload is executed by setup.
param(
    [ValidateSet("cuda", "cpu")]
    [string]$Lane = "cuda"
)

$ErrorActionPreference = "Stop"
$IkProjectRoot = Split-Path -Parent $PSScriptRoot
$IkEnvironment = Join-Path $IkProjectRoot ".venv-ik"
if (Test-Path -LiteralPath $IkEnvironment) {
    throw "Refusing to modify existing .venv-ik. Use a fresh checkout for another runtime identity."
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python 3.12 with the py launcher is required."
}

& py -3.12 -m venv $IkEnvironment
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 virtual environment creation failed." }
$IkPython = Join-Path $IkEnvironment "Scripts\python.exe"
& $IkPython -m pip install setuptools==84.0.0 wheel==0.45.1 numpy==2.3.5
if ($LASTEXITCODE -ne 0) { throw "Pinned build tools and NumPy installation failed." }
if ($Lane -eq "cuda") {
    $IkTorchVersion = "2.8.0+cu126"
    $IkWheelIndex = "https://download.pytorch.org/whl/cu126"
} else {
    $IkTorchVersion = "2.8.0+cpu"
    $IkWheelIndex = "https://download.pytorch.org/whl/cpu"
}
& $IkPython -m pip install "torch==$IkTorchVersion" --index-url $IkWheelIndex
if ($LASTEXITCODE -ne 0) { throw "Pinned Torch installation failed." }
& $IkPython -m pip install --no-build-isolation -e $IkProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }
& $IkPython -c "import sys; from importlib.metadata import version; assert sys.version_info[:2] == (3, 12); assert version('numpy') == '2.3.5'; assert version('torch') == sys.argv[1]; print('Optional IK package versions verified; device qualification has not run.')" $IkTorchVersion
if ($LASTEXITCODE -ne 0) { throw "Installed runtime identity differs from the requested lane." }

Write-Output "Setup complete. Use .venv-ik\Scripts\python.exe for the isolated runtime."
Write-Output "Read docs/IK_RUNTIME.md before declaring or executing a bounded smoke run."
