#Requires -Version 5.1
<#
  VancoDose launcher for Windows. Run through "Start VancoDose.bat" (double-click).
  Makes sure 64-bit Python 3.12-3.14 is installed (installs Python 3.12 for the current user
  if needed, no administrator rights required), adds a VancoDose shortcut with the app icon to
  the Desktop, then hands over to scripts\launcher.py, which installs packages and starts the app.
#>
param([switch]$NoShortcut)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
try { $Host.UI.RawUI.WindowTitle = 'VancoDose' } catch { }
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$PyVersion = '3.12.10'
$LogDir = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir 'launcher.log'

function Say([string]$Message, [string]$Color = 'Gray') {
  Write-Host $Message -ForegroundColor $Color
  Add-Content -LiteralPath $Log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $Message) -Encoding UTF8
}
function Fail([string]$Message) { Say ''; Say "X $Message" 'Red' }

function Test-Python([string]$Exe) {
  # 64-bit x64 Python 3.12-3.14 (x64 also on ARM PCs: some packages have no ARM build for Windows)
  if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) { return $false }
  if ($Exe -like '*\WindowsApps\*') { return $false }   # Microsoft Store placeholder, not a real Python
  try {
    & $Exe -c "import sys, platform; sys.exit(0 if (3,12) <= sys.version_info[:2] <= (3,14) and platform.machine() in ('AMD64','x86_64') else 1)" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
  } catch { return $false }
}

function Find-Python {
  $candidates = New-Object System.Collections.Generic.List[string]
  if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in '3.12', '3.13', '3.14') {
      try { $p = (& py "-$v" -c "import sys; print(sys.executable)" 2>$null); if ($LASTEXITCODE -eq 0 -and $p) { $candidates.Add($p.Trim()) } } catch { }
    }
  }
  foreach ($v in '312', '313', '314') {
    $candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"))
    $candidates.Add((Join-Path $env:ProgramFiles "Python$v\python.exe"))
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { $candidates.Add($cmd.Source) }
  foreach ($c in $candidates) { if (Test-Python $c) { return $c } }
  return $null
}

function Install-Python {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Say 'Installing Python 3.12 with Windows Package Manager (winget)...'
    try {
      & winget install --id Python.Python.3.12 --exact --scope user --architecture x64 --silent `
        --accept-package-agreements --accept-source-agreements --disable-interactivity | Out-Host
      if ($LASTEXITCODE -eq 0 -or (Find-Python)) { return $true }
    } catch { }
    Say 'winget could not install Python; trying the official installer instead.' 'Yellow'
  }
  $url = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-amd64.exe"
  $file = Join-Path $env:TEMP "python-$PyVersion-amd64.exe"
  Say "Downloading Python $PyVersion from python.org (about 25 MB)..."
  $ProgressPreference = 'SilentlyContinue'   # the progress bar makes downloads very slow in Windows PowerShell 5.1
  try { Invoke-WebRequest -Uri $url -OutFile $file -UseBasicParsing }
  catch { Fail 'The download failed. Check your internet connection (or proxy) and try again.'; return $false }
  $sig = Get-AuthenticodeSignature -LiteralPath $file
  if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
    Fail 'The downloaded installer is not signed by the Python Software Foundation; it was not run.'; return $false
  }
  Say 'Installing Python for your user account (no administrator rights needed)...'
  $p = Start-Process -FilePath $file -Wait -PassThru -ArgumentList '/quiet', 'InstallAllUsers=0', 'PrependPath=1', 'Include_launcher=1', 'Include_test=0', 'Shortcuts=0'
  Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue
  return ($p.ExitCode -eq 0)
}

function Add-Shortcut {
  try {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $lnk = Join-Path $desktop 'VancoDose.lnk'
    if (Test-Path -LiteralPath $lnk) { return }
    $shell = New-Object -ComObject WScript.Shell
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath = Join-Path $Root 'Start VancoDose.bat'
    $s.WorkingDirectory = $Root
    $s.IconLocation = (Join-Path $Root 'assets\vancodose.ico') + ',0'
    $s.Description = 'Start VancoDose (vancomycin precision dosing)'
    $s.Save()
    Say 'Added a VancoDose shortcut with the app icon to your Desktop.' 'Green'
  } catch { Say 'Could not add a Desktop shortcut (not required).' 'Yellow' }
}

# ------------------------------------------------------------------------------------------
# Dot-sourcing (. .\scripts\start-vancodose.ps1) loads the functions only; used by the automated Windows test.
if ($MyInvocation.InvocationName -ne '.') {
  $py = Find-Python
  if (-not $py) {
    Say 'Python 3.12 or newer (64-bit) was not found on this computer. Installing it now...' 'Yellow'
    if (-not (Install-Python)) {
      Say 'You can install Python yourself from https://www.python.org/downloads/windows/'
      Say '(choose "Windows installer (64-bit)" and tick "Add python.exe to PATH"), then run this launcher again.'
      exit 3
    }
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $py = Find-Python
    if (-not $py) { Fail 'Python was installed but could not be found. Restart the computer and run this launcher again.'; exit 3 }
  }
  if (-not $NoShortcut) { Add-Shortcut }

  & $py (Join-Path $Root 'scripts\launcher.py') @args
  exit $LASTEXITCODE
}
