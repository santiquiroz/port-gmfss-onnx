# Crea el venv CPU-only del toolkit de referencia GMFSS_Fortuna (py3.11).
# Uso: pwsh -File toolkit/setup-env.ps1
$ErrorActionPreference = 'Continue'
$repo = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $repo '.venv'

if (-not (Test-Path $venv)) {
    py -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$python = Join-Path $venv 'Scripts\python.exe'

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

& $python -c "import torch; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "torch import failed" }

& $python -c "import cupy" 2>$null
if ($LASTEXITCODE -eq 0) { throw "cupy must NOT be installed in this environment (CPU-only reference toolkit)" }

& $python -c "import onnx, onnxruntime, cv2, skimage, numpy; print('deps OK')"
if ($LASTEXITCODE -ne 0) { throw "dependency import failed" }

Write-Host 'Environment ready.'
