# Reel Studio AI updater: downloads the latest app code from GitHub.
# Needs no Git install. Only code files are replaced; projects, videos and
# settings are never touched. Any failure just starts the current version.

$ErrorActionPreference = 'Stop'
$AppDir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo        = 'arsalansirkhot/reelstudioai'
$Branch      = 'main'
$TokenFile   = Join-Path $AppDir 'github_token.txt'
$VersionFile = Join-Path $AppDir 'installed_version.txt'
# Files copied from GitHub into this folder on every update
$Files = @('app.py', 'requirements.txt', 'README.txt', 'update.ps1', '.streamlit/config.toml')

try {
    if (-not (Test-Path $TokenFile)) {
        Write-Host ''
        Write-Host 'To get automatic updates, paste your GitHub read-only key below.'
        $key = Read-Host 'GitHub key (or just press Enter to skip)'
        if ([string]::IsNullOrWhiteSpace($key)) { Write-Host 'Skipping update check.'; exit 0 }
        Set-Content -Path $TokenFile -Value $key.Trim() -NoNewline
        Write-Host 'Key saved.'
    }
    $token = (Get-Content $TokenFile -Raw).Trim()

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $headers = @{
        Authorization          = "Bearer $token"
        'User-Agent'           = 'ReelStudioAI-updater'
        'X-GitHub-Api-Version' = '2022-11-28'
    }

    $sha = (Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/commits/$Branch" -Headers $headers -TimeoutSec 20).sha
    $short = $sha.Substring(0, 7)
    $current = ''
    if (Test-Path $VersionFile) { $current = (Get-Content $VersionFile -Raw).Trim() }
    if ($sha -eq $current) { Write-Host "App is up to date (version $short)."; exit 0 }

    Write-Host "Downloading update (version $short)..."
    $tmp = Join-Path $env:TEMP ('reelstudio_update_' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Join-Path $tmp 'src.zip'
        Invoke-WebRequest -Uri "https://api.github.com/repos/$Repo/zipball/$sha" -Headers $headers -OutFile $zip -UseBasicParsing -TimeoutSec 180
        Expand-Archive -Path $zip -DestinationPath $tmp
        $src = (Get-ChildItem -Path $tmp -Directory | Select-Object -First 1).FullName

        foreach ($f in $Files) {
            $from = Join-Path $src $f
            if (Test-Path $from) {
                $to = Join-Path $AppDir $f
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $to) | Out-Null
                Copy-Item -Path $from -Destination $to -Force
            }
        }
        # The launcher can't overwrite itself while running, so it swaps this in.
        $bat = Join-Path $src 'run_windows.bat'
        if (Test-Path $bat) { Copy-Item -Path $bat -Destination (Join-Path $AppDir 'run_windows.bat.new') -Force }

        Set-Content -Path $VersionFile -Value $sha -NoNewline
        Write-Host "Update installed (version $short)."
    }
    finally {
        Remove-Item -Path $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
catch {
    $msg = $_.Exception.Message
    if ($msg -match '401|403|404') {
        Write-Host 'GitHub refused the key. Delete github_token.txt and paste a new key next time.'
    } else {
        Write-Host "Could not check for updates ($msg)."
    }
    Write-Host 'Starting with the version already on this computer.'
}
exit 0
