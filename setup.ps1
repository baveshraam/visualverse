# VisualVerse - GPU Environment Setup Script
# RTX 4060, CUDA 12.1
# Usage: powershell -ExecutionPolicy Bypass -File .\setup.ps1

$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) { Write-Host "" ; Write-Host "== $msg ==" -ForegroundColor Cyan }
function Write-OK([string]$msg)   { Write-Host "  OK  $msg" -ForegroundColor Green  }
function Write-Fail([string]$msg) { Write-Host "  ERR $msg" -ForegroundColor Red    }
function Write-Warn([string]$msg) { Write-Host "  WRN $msg" -ForegroundColor Yellow }

# --- Step 0: Python check ---
Write-Step "Checking Python installation"
try {
    $v = python --version 2>&1
    Write-OK "Found: $v"
} catch {
    Write-Fail "Python not found. Install Python 3.10+ and add to PATH."
    exit 1
}

# --- Step 1: Upgrade pip ---
Write-Step "Upgrading pip"
python -m pip install --upgrade pip | Out-Null
Write-OK "pip upgraded"

# --- Step 2: PyTorch 2.4.0 with CUDA 12.1 ---
Write-Step "Installing PyTorch 2.6.0 + CUDA 12.1 (~2.5 GB download)"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Fail "PyTorch install failed. Check your network."
    exit 1
}
Write-OK "PyTorch 2.6.0 + CUDA 12.1 installed"


# --- Step 3: Core ML libs ---
Write-Step "Installing transformers, bitsandbytes, accelerate, diffusers"

# sentencepiece must be installed separately - it requires C++ build tools if no wheel exists
# Using --only-binary to force pre-built wheel download
pip install sentencepiece==0.2.0 --only-binary :all:
if ($LASTEXITCODE -ne 0) {
    # Fallback: install without --only-binary (needs MSVC on Windows)
    Write-Warn "Pre-built sentencepiece wheel not found, trying source build..."
    pip install sentencepiece==0.2.0
}

pip install "transformers>=4.43.0" "bitsandbytes>=0.43.0" "accelerate>=0.30.0" "diffusers>=0.29.0" "tokenizers>=0.19.0"
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Core ML library install failed."
    exit 1
}
Write-OK "Core ML libs installed"


# --- Step 4: SpaCy + langdetect ---
Write-Step "Installing SpaCy + langdetect"
pip install "spacy>=3.8.7" langdetect==1.0.9
if ($LASTEXITCODE -ne 0) {
    Write-Fail "SpaCy / langdetect install failed."
    exit 1
}
Write-OK "SpaCy + langdetect installed"

# --- Step 5: Remaining backend requirements ---
Write-Step "Installing remaining packages from backend/requirements.txt"
$reqFile = Join-Path $PSScriptRoot "backend\requirements.txt"

if (Test-Path $reqFile) {
    pip install -r $reqFile --extra-index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Some packages had errors - check output above."
    } else {
        Write-OK "All requirements installed"
    }
} else {
    Write-Warn "requirements.txt not found at $reqFile - skipping"
}

# --- Step 6: SpaCy English model ---
Write-Step "Downloading SpaCy en_core_web_sm"
python -m spacy download en_core_web_sm
if ($LASTEXITCODE -ne 0) {
    Write-Fail "SpaCy model download failed."
    exit 1
}
Write-OK "SpaCy en_core_web_sm ready"

# --- Step 7: GPU Verification ---
Write-Step "Verifying GPU access"

$gpuScript = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.py'

$pythonLines = @(
    "import torch, sys",
    "print('Python    :', sys.version.split()[0])",
    "print('PyTorch   :', torch.__version__)",
    "cuda_ok = torch.cuda.is_available()",
    "print('CUDA avail:', cuda_ok)",
    "if cuda_ok:",
    "    name = torch.cuda.get_device_name(0)",
    "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9",
    "    print('GPU       :', name)",
    "    print('VRAM total:', str(round(vram, 1)) + ' GB')",
    "    t = torch.zeros(1, device='cuda')",
    "    del t",
    "    torch.cuda.empty_cache()",
    "    alloc = torch.cuda.memory_allocated() / 1e9",
    "    print('VRAM used :', str(round(alloc, 2)) + ' GB')",
    "    print('GPU compute: OK')",
    "else:",
    "    print('WARNING: CUDA not available - check driver and CUDA wheel')",
    "    import sys; sys.exit(1)"
)

[System.IO.File]::WriteAllLines($gpuScript, $pythonLines, [System.Text.Encoding]::UTF8)

python $gpuScript
$gpuExit = $LASTEXITCODE
Remove-Item $gpuScript -Force -ErrorAction SilentlyContinue

if ($gpuExit -ne 0) {
    Write-Fail "GPU verification failed. Check NVIDIA drivers."
    exit 1
}
Write-OK "GPU verified successfully"

# --- Done ---
$line = "=" * 60
Write-Host ""
Write-Host $line -ForegroundColor Green
Write-Host " VisualVerse environment is ready!" -ForegroundColor Green
Write-Host ""
Write-Host " Next steps:" -ForegroundColor White
Write-Host "   1. cd backend" -ForegroundColor DarkGray
Write-Host "   2. python -m uvicorn main:app --reload --port 8000" -ForegroundColor DarkGray
Write-Host "   3. Open http://localhost:8000/docs" -ForegroundColor DarkGray
Write-Host $line -ForegroundColor Green
