<#
.SYNOPSIS
  Keeps the remote training chain alive across Paperspace instance deaths.

.DESCRIPTION
  The recovery has to run from this machine: when the free-tier instance dies,
  only the local jlab CLI can bring a new one up. Anything running on the remote
  dies with it, so a watchdog there cannot help.

  Every cycle this script asks the remote to run auto_resume.py, which:
    - restarts the sidepanel publisher if it died,
    - does nothing if a job is already training,
    - marks a finished arm done and starts the next one,
    - otherwise resumes the current arm from its newest checkpoint, with the
      best accuracy so far as the floor so a resume cannot overwrite
      best_model.pt with a worse value.

  If the remote does not answer, the script works through the failure modes seen
  in practice, in increasing order of disruption: restart the kernel session,
  then run jlab setup to bring up a new instance.

.PARAMETER IntervalSeconds
  Seconds between cycles. Default 300. A death costs at most this plus whatever
  is between the last checkpoint and the crash.

.PARAMETER MaxHours
  Stop after this many hours. Default 24. Use 0 to run until the chain reports
  every job finished.

.EXAMPLE
  pwsh -File scripts\auto_renew.ps1
  pwsh -File scripts\auto_renew.ps1 -IntervalSeconds 180 -MaxHours 48
#>
[CmdletBinding()]
param(
  [int]$IntervalSeconds = 300,
  [double]$MaxHours = 24,
  [string]$LogPath = "$PSScriptRoot\..\auto_renew.log"
)

$ErrorActionPreference = "Continue"
$deadline = if ($MaxHours -gt 0) { (Get-Date).AddHours($MaxHours) } else { [DateTime]::MaxValue }
# auto_resume.py prints exactly one of these when it is reached and healthy.
$healthy = 'JOB_ALIVE|RESUMING|STARTING|MARKED|ALL_JOBS_DONE'

function Write-Line([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $LogPath -Value $line
}

function Invoke-Resume {
  # Returns the remote output, or $null if the remote could not be reached.
  $out = & jlab exec "python /notebooks/auto_resume.py" 2>&1 | Out-String
  if ($out -match $healthy) { return $out }
  return $null
}

Write-Line "auto_renew starting, every ${IntervalSeconds}s, stopping $(if ($MaxHours -gt 0) { $deadline } else { 'when all jobs finish' })"

$cycle = 0
while ((Get-Date) -lt $deadline) {
  $cycle++
  $out = Invoke-Resume

  if ($null -eq $out) {
    # A wedged or missing kernel is cheap to fix and does not disturb training,
    # so try that before assuming the instance is gone.
    Write-Line "no answer, restarting kernel session"
    & jlab session stop  *> $null
    & jlab session start *> $null
    $out = Invoke-Resume
  }

  if ($null -eq $out) {
    Write-Line "still no answer, bringing up a new instance"
    & jlab setup *> $null
    $out = Invoke-Resume
  }

  if ($null -eq $out) {
    Write-Line "recovery failed this cycle, will retry"
  } else {
    $summary = ($out -split "`n" | Where-Object { $_ -match $healthy } | Select-Object -First 1).Trim()
    Write-Line $summary
    if ($summary -match 'ALL_JOBS_DONE') {
      Write-Line "chain complete after $cycle cycles"
      break
    }
  }

  Start-Sleep -Seconds $IntervalSeconds
}

Write-Line "auto_renew exiting after $cycle cycles"
