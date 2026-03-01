param(
  [int]$TargetPid,
  [string]$OutLog,
  [string]$MonitorLog
)
$ErrorActionPreference = 'SilentlyContinue'
Add-Content -Path $MonitorLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] monitor started pid=$TargetPid"
while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
  $last = ''
  if (Test-Path $OutLog) { $last = (Get-Content $OutLog -Tail 1) }
  Add-Content -Path $MonitorLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] running pid=$TargetPid last=$last"
  Start-Sleep -Seconds 30
}
Add-Content -Path $MonitorLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] finished pid=$TargetPid"
