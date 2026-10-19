$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required." }
& .\.venv\Scripts\python.exe -m pip install setuptools==84.0.0 wheel==0.45.1
if ($LASTEXITCODE -ne 0) { throw "Build tools failed to install." }
& .\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }
& .\.venv\Scripts\python.exe -m nca doctor
if ($LASTEXITCODE -ne 0) { throw "Environment report failed." }
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
if ($LASTEXITCODE -ne 0) { throw "CPU conformance failed." }
