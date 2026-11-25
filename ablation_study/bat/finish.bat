@echo off
rem Last block of COMMANDS.md: put the sleep timeout back to 5 minutes on AC power.
powercfg /change standby-timeout-ac 5
if errorlevel 1 (
  echo [WARN] Could not change the sleep timeout. Set it by hand or run as Administrator.
) else (
  echo Sleep timeout on AC power set back to 5 minutes.
)
echo "%cmdcmdline%" | find /i "/c" >nul && pause
