# Put this project's toolchain on PATH (Windows):  . .\scripts\env.ps1
#
# The PowerShell counterpart of scripts/env.sh (docs/TESTING.md T-W9). Dot-source
# it -- running it as a script gets its own scope and the changes are discarded.
#
# rustup installs cargo to %USERPROFILE%\.cargo\bin and uv installs to
# %USERPROFILE%\.local\bin; neither is guaranteed to be on PATH in a fresh
# shell. VS Code users get the same thing from .vscode/settings.json without
# sourcing anything.

$repo = Split-Path -Parent $PSScriptRoot

foreach ($dir in @("$env:USERPROFILE\.cargo\bin", "$env:USERPROFILE\.local\bin")) {
    if ((Test-Path $dir) -and ($env:PATH -split ';' -notcontains $dir)) {
        $env:PATH = "$dir;$env:PATH"
    }
}

# online-py builds against pyo3's abi3-py312, so the Rust build needs a >= 3.12
# interpreter. `uv run cargo ...` handles this via VIRTUAL_ENV; for a bare
# `cargo ...` this makes the project venv the one pyo3 finds. Note the layout
# difference from Unix: .venv\Scripts\python.exe, not .venv/bin/python.
$py = Join-Path $repo '.venv\Scripts\python.exe'
if (Test-Path $py) { $env:PYO3_PYTHON = $py }

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Warning 'scripts/env.ps1: cargo not found; install Rust from https://rustup.rs'
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Warning 'scripts/env.ps1: uv not found; see https://docs.astral.sh/uv/'
}
