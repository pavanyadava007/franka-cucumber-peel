#!/bin/bash
# Keep Windows from going to sleep while a process runs. This PC sleeps after 10 minutes without input, which pauses
# WSL and every job in it. Uses SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED): the screen may still
# turn off, settings are not changed, and normal sleep returns as soon as this ends (process gone, or 16 h at most).
#   bash scripts/dev/keep_awake.sh <pid>
PID=$1
FLAG="$(cd "$(dirname "$0")/../.." && pwd)/runs/keep_awake.flag"
touch "$FLAG"
WINFLAG="\\\\wsl.localhost\\${WSL_DISTRO_NAME}$(echo "$FLAG" | sed 's|/|\\|g')"
PS="Add-Type -Namespace W -Name P -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'
\$end = (Get-Date).AddHours(16)
while ((Test-Path -LiteralPath '$WINFLAG') -and ((Get-Date) -lt \$end)) { [void][W.P]::SetThreadExecutionState(0x80000001); Start-Sleep 60 }
[void][W.P]::SetThreadExecutionState(0x80000000)"
(cd /mnt/c && exec powershell.exe -NoProfile -EncodedCommand "$(printf '%s' "$PS" | iconv -t utf-16le | base64 -w0)") &
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
rm -f "$FLAG"
