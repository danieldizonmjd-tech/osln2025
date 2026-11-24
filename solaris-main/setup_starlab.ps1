$ErrorActionPreference = "Stop"

Write-Host "Setting up Starlab Solar Energy Monitor..." -ForegroundColor Cyan

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv package manager..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv python install 3.12
uv venv --python 3.12 .venv

uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
uv pip install --python .\.venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "Initializing database..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -c "from database import init_database; init_database(); print('Database ready: starlab_energy.db')"

Write-Host "Training CUDA-capable anomaly model..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe train_model.py --samples 52000 --epochs 28 --batch-size 256

Write-Host ""
Write-Host "Setup complete. Launch with:" -ForegroundColor Green
Write-Host '.\.venv\Scripts\python.exe predict.py' -ForegroundColor Green
