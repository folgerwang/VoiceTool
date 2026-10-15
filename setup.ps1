# VoiceTool setup - everything say.py needs, inside this folder.
#
#   venv\                 Python 3.10 (CosyVoice's own target; 3.13+ lacks wheels)
#   third_party\CosyVoice the inference code (not on PyPI) + Matcha-TTS submodule
#   models\cosyvoice2     the weights (copied from a sibling checkout if one has
#                         them, else downloaded from Hugging Face, ~4.6 GB)
#   voices\               your actors: name.wav + name.txt pairs
#
# Re-runnable: every step checks before it does anything.
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1            # all steps
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Model v3  # also fetch CosyVoice3

param(
    [ValidateSet("v2", "v3")][string]$Model = "v2",
    [switch]$Cpu
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV = Join-Path $HERE "venv"
$PY = Join-Path $VENV "Scripts\python.exe"

function Say($s) { Write-Host "  $s" }

# --- [1/5] python 3.10 venv ------------------------------------------------
if (-not (Test-Path $PY)) {
    $cands = @(
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python310\python.exe"
    )
    $base = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $base) {
        $pyl = Get-Command py -ErrorAction SilentlyContinue
        if ($pyl) {
            $p = & py -3.10 -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { $base = $p.Trim() }
        }
    }
    if (-not $base) {
        Write-Host "  x Python 3.10 not found. Install it from https://www.python.org/downloads/release/python-31011/"
        Write-Host "    (CosyVoice's dependencies have no wheels for 3.13+)."
        exit 1
    }
    Say "[1/5] venv from $base"
    & $base -m venv $VENV
    if (-not (Test-Path $PY)) { Write-Host "  x venv creation failed"; exit 1 }
} else {
    Say "[1/5] venv exists"
}
& $PY -m pip install -q -U pip 2>&1 | Out-Null

# --- [2/5] torch ------------------------------------------------------------
& $PY -c "import torch, torchaudio" 2>$null
$needTorch = ($LASTEXITCODE -ne 0)
if (-not $needTorch -and -not $Cpu) {
    & $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    $needTorch = ($LASTEXITCODE -ne 0)
}
if ($needTorch) {
    if ($Cpu) {
        Say "[2/5] torch (cpu build)"
        & $PY -m pip install -q torch torchaudio 2>&1 | Select-Object -Last 2
    } else {
        # The PyPI wheel is CPU-only on Windows. CUDA builds come from PyTorch's
        # own index, one per CUDA series; try the current ones newest-first and
        # keep the first that actually sees the GPU. cu128+ is what an RTX 50xx
        # (Blackwell) needs.
        $done = $false
        foreach ($cu in @("cu128", "cu130", "cu126")) {
            Say "[2/5] torch from $cu ..."
            & $PY -m pip uninstall -q -y torch torchaudio 2>&1 | Out-Null
            & $PY -m pip install -q torch torchaudio --index-url "https://download.pytorch.org/whl/$cu" 2>&1 | Select-Object -Last 2
            if ($LASTEXITCODE -ne 0) { continue }
            & $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { $done = $true; break }
        }
        if (-not $done) {
            Say "!  no CUDA torch worked - falling back to the CPU build (slow, but it runs)"
            & $PY -m pip install -q torch torchaudio 2>&1 | Select-Object -Last 2
        }
    }
}
$tv = & $PY -c "import torch;print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
Say "[2/5] torch $tv"

# --- [3/5] CosyVoice code ---------------------------------------------------
$TP = Join-Path $HERE "third_party\CosyVoice"
if (-not (Test-Path (Join-Path $TP "cosyvoice\cli\cosyvoice.py"))) {
    Say "[3/5] cloning CosyVoice (+ Matcha-TTS submodule) ..."
    New-Item -ItemType Directory -Force (Join-Path $HERE "third_party") | Out-Null
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice $TP
    if (-not (Test-Path (Join-Path $TP "cosyvoice\cli\cosyvoice.py"))) { Write-Host "  x clone failed (is git installed?)"; exit 1 }
} else {
    Say "[3/5] CosyVoice checkout exists"
}
if (-not (Test-Path (Join-Path $TP "third_party\Matcha-TTS\matcha"))) {
    Say "[3/5] fetching the Matcha-TTS submodule ..."
    Push-Location $TP; git submodule update --init --recursive; Pop-Location
}

# --- [4/5] python deps ------------------------------------------------------
$env:PIP_CONSTRAINT = Join-Path $HERE "pip-constraints.txt"
Say "[4/5] CosyVoice inference deps ..."
& $PY -m pip install -q -r (Join-Path $HERE "requirements.txt") 2>&1 | Select-Object -Last 3
$env:PIP_CONSTRAINT = ""
& $PY -c "import hyperpyyaml, whisper, onnxruntime, diffusers, conformer, wetext, soundfile, imageio_ffmpeg" 2>$null
if ($LASTEXITCODE -ne 0) { Say "!  some deps did not import - say.py will tell you which" } else { Say "[4/5] deps ok" }

# --- [5/5] weights ----------------------------------------------------------
$MODELS = Join-Path $HERE "models"
New-Item -ItemType Directory -Force $MODELS | Out-Null
function Have-Model($dir, $yaml) { Test-Path (Join-Path (Join-Path $MODELS $dir) $yaml) }

if (-not (Have-Model "cosyvoice2" "cosyvoice2.yaml")) {
    # A sibling Bookreel checkout may already hold the 4.6 GB; copy rather than download.
    $sib = Get-ChildItem (Split-Path -Parent $HERE) -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "models\cosyvoice2" } |
        Where-Object { Test-Path (Join-Path $_ "cosyvoice2.yaml") } | Select-Object -First 1
    if ($sib) {
        Say "[5/5] copying cosyvoice2 from $sib ..."
        robocopy $sib (Join-Path $MODELS "cosyvoice2") /E /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null
    } else {
        Say "[5/5] downloading FunAudioLLM/CosyVoice2-0.5B (~4.6 GB) ..."
        & $PY -m pip install -q "huggingface_hub[cli]" 2>&1 | Out-Null
        & $PY -m huggingface_hub.commands.huggingface_cli download FunAudioLLM/CosyVoice2-0.5B --local-dir (Join-Path $MODELS "cosyvoice2")
    }
}
if (Have-Model "cosyvoice2" "cosyvoice2.yaml") { Say "[5/5] models\cosyvoice2 ok" } else { Say "!  models\cosyvoice2 missing" }

if ($Model -eq "v3" -and -not (Have-Model "cosyvoice3" "cosyvoice3.yaml")) {
    Say "[5/5] downloading FunAudioLLM/Fun-CosyVoice3-0.5B-2512 ..."
    & $PY -m pip install -q "huggingface_hub[cli]" 2>&1 | Out-Null
    & $PY -m huggingface_hub.commands.huggingface_cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --local-dir (Join-Path $MODELS "cosyvoice3")
    if (Have-Model "cosyvoice3" "cosyvoice3.yaml") { Say "[5/5] models\cosyvoice3 ok (say.py prefers it over v2)" }
}

Write-Host ""
Write-Host "  done.   .\say --list      .\say -v storyteller ""要念的话""      .\say --web"
exit 0
