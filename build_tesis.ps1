param(
    [string]$TexFile = "main_tesis.tex",
    [string]$OutDir = "build",
    [switch]$ShellEscape,
    [switch]$CleanAux,
    [switch]$UseLatexmk
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TexFile)) {
    throw "No se encontro el archivo TeX: $TexFile"
}

$texPath = Resolve-Path -LiteralPath $TexFile
$workDir = Split-Path -Parent $texPath
$texName = Split-Path -Leaf $texPath
$texBase = [System.IO.Path]::GetFileNameWithoutExtension($texName)

if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
$outDirAbs = Resolve-Path -LiteralPath $OutDir

$flags = @("-interaction=nonstopmode", "-file-line-error", "-halt-on-error", "-synctex=1")
if ($ShellEscape) {
    $flags += "-shell-escape"
}

function Run-Step {
    param(
        [string]$Command,
        [string[]]$Args
    )
    Write-Host ">> $Command $($Args -join ' ')"
    & $Command @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Fallo: $Command (exit code $LASTEXITCODE)"
    }
}

Push-Location $workDir
try {
    $usedLatexmk = $false

    if ($UseLatexmk) {
        if (-not (Get-Command latexmk -ErrorAction SilentlyContinue)) {
            throw "Se pidio -UseLatexmk pero latexmk no esta disponible en PATH."
        }
        $mkArgs = @("-xelatex", "-outdir=$outDirAbs", "-pdf") + $flags + @($texName)
        Run-Step -Command "latexmk" -Args $mkArgs
        $usedLatexmk = $true
    }

    if (-not $usedLatexmk -and (Get-Command xelatex -ErrorAction SilentlyContinue)) {
        $xeArgs = @("-output-directory=$outDirAbs") + $flags + @($texName)
        Run-Step -Command "xelatex" -Args $xeArgs
        Run-Step -Command "xelatex" -Args $xeArgs
        Run-Step -Command "xelatex" -Args $xeArgs
    }
    elseif (-not $usedLatexmk -and (Get-Command lualatex -ErrorAction SilentlyContinue)) {
        $luaArgs = @("-output-directory=$outDirAbs") + $flags + @($texName)
        Run-Step -Command "lualatex" -Args $luaArgs
        Run-Step -Command "lualatex" -Args $luaArgs
        Run-Step -Command "lualatex" -Args $luaArgs
    }
    elseif (-not $usedLatexmk -and (Get-Command latexmk -ErrorAction SilentlyContinue)) {
        $mkArgs = @("-xelatex", "-outdir=$outDirAbs", "-pdf") + $flags + @($texName)
        Run-Step -Command "latexmk" -Args $mkArgs
        $usedLatexmk = $true
    }
    elseif (-not $usedLatexmk) {
        throw "No se encontro latexmk, xelatex ni lualatex en PATH."
    }

    $pdfPath = Join-Path $outDirAbs "$texBase.pdf"
    if (-not (Test-Path -LiteralPath $pdfPath)) {
        throw "La compilacion termino pero no se encontro el PDF: $pdfPath"
    }

    if ($CleanAux) {
        if ($usedLatexmk -and (Get-Command latexmk -ErrorAction SilentlyContinue)) {
            Run-Step -Command "latexmk" -Args @("-outdir=$outDirAbs", "-c", $texName)
        }
        else {
            Get-ChildItem -LiteralPath $outDirAbs -File |
                Where-Object { $_.Extension -in @(".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls", ".log", ".nav", ".out", ".run.xml", ".snm", ".synctex.gz", ".toc", ".lot", ".lof") } |
                Remove-Item -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host ""
    Write-Host "PDF generado correctamente:"
    Write-Host $pdfPath
}
finally {
    Pop-Location
}
