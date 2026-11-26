@echo off
rem queue.bat RUN_BAT [RUN_BAT ...]
rem Runs the given run_NN_*.bat files one after another, unattended (overnight).
rem   - never waits for a key between runs
rem   - a failed run is logged and the queue moves on to the next one
rem   - the full console output of each run is saved:
rem       finished run -> results\<recipe>_ep150\console_log.txt
rem       failed run   -> results\_console\<recipe>.txt
rem   - one summary line per run goes to results\_console\queue_<computer name>.txt
rem Example:  queue.bat run_06_x_slope_nosynth_s1 run_07_x_slope_nocost_s1
rem Normally you double-click queue_A.bat / queue_B.bat / queue_C.bat / queue_D.bat instead.
setlocal EnableExtensions
set "NOPAUSE=1"
set "HERE=%~dp0"
call "%HERE%env.bat"
if errorlevel 1 goto :done
if not exist "results\_console" mkdir "results\_console"
set "QLOG=results\_console\queue_%COMPUTERNAME%.txt"
echo [%date% %time%] queue started on %COMPUTERNAME%: %*>> "%QLOG%"

:next
if "%~1"=="" goto :done
set "RUN=%~n1"
rem run_06_x_slope_nosynth_s1 -> x_slope_nosynth_s1
set "CFG=%RUN:~7%"
set "CON=results\_console\%CFG%.txt"
echo.
echo [%date% %time%] START %RUN%   (live log: %CD%\%CON%)
echo [%date% %time%] START %RUN%>> "%QLOG%"
call "%HERE%%RUN%.bat" > "%CON%" 2>&1
set "RC=%errorlevel%"
if "%RC%"=="0" if exist "results\%CFG%_ep150\" move /y "%CON%" "results\%CFG%_ep150\console_log.txt" >nul
if "%RC%"=="0" (set "WHERE=results\%CFG%_ep150\console_log.txt") else (set "WHERE=%CON%")
echo [%date% %time%] END   %RUN%  exit=%RC%  log=%WHERE%
echo [%date% %time%] END   %RUN%  exit=%RC%  log=%WHERE%>> "%QLOG%"
shift
goto :next

:done
echo [%date% %time%] queue finished>> "%QLOG%"
echo.
echo Queue finished. Summary: %CD%\%QLOG%
type "%QLOG%"
call "%HERE%finish.bat"
pause
