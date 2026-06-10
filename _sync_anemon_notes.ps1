$ErrorActionPreference = "Continue"

$Root = "C:\Users\Clezv\Documents\Anemon"
$Markdown = Join-Path $Root "_remote_anemon_notes_cte_monitor.md"
$Log = Join-Path $Root "_sync_anemon_notes.log"
$Url = $env:ANEMON_NOTES_URL
if (-not $Url) {
  $Url = "https://viz-qcc-production.up.railway.app/api/notes"
}

function Write-Log($Message) {
  $stamp = [DateTimeOffset]::Now.ToString("yyyy-MM-dd HH:mm:ss zzz")
  Add-Content -Path $Log -Value "[$stamp] $Message" -Encoding UTF8
}

Write-Log "starting notes publisher url=$Url"

while ($true) {
  try {
    Push-Location $Root
    jlab download Manta/docs/anemon_notes_cte_monitor.md $Markdown | Out-Null
    Pop-Location

    if (-not (Test-Path $Markdown)) {
      Write-Log "markdown missing after download"
      Start-Sleep -Seconds 15
      continue
    }

    $token = $env:ANEMON_PUBLISH_TOKEN
    if (-not $token) {
      Write-Log "ANEMON_PUBLISH_TOKEN is not set; cannot POST to /api/notes"
      Start-Sleep -Seconds 15
      continue
    }

    $body = Get-Content -Raw -Path $Markdown
    $payload = [ordered]@{
      title = "CNXXL-Quat CTE Monitor"
      body = $body
    } | ConvertTo-Json -Depth 4

    $headers = @{
      Authorization = "Bearer $token"
      "Content-Type" = "application/json; charset=utf-8"
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $response = Invoke-RestMethod -Method Post -Uri $Url -Headers $headers -Body $bytes -TimeoutSec 20
    Write-Log "posted notes ok bytes=$($response.bytes) chars=$($body.Length)"
  } catch {
    try { Pop-Location } catch {}
    $detail = $_.Exception.Message
    if ($_.Exception.Response) {
      try {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $detail = "$detail body=$($reader.ReadToEnd())"
      } catch {}
    }
    Write-Log "post failed: $detail"
  }

  Start-Sleep -Seconds 15
}
