param(
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
}

& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
& '.\.venv\Scripts\python.exe' -m pip install -e .

& '.\.venv\Scripts\python.exe' -c "import torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
