@echo off
rem _run.bat CFG [BASE_RUN ...]
rem Trains one recipe, writes honest_curve.txt, then compares it with each BASE_RUN in turn.
rem Optional: set NEEDS=<teacher run> first; that run's models.pkl must exist.
rem Never overwrites an existing results folder. Stops at the first failure.
setlocal EnableExtensions
set "HERE=%~dp0"
set "CFG=%~1"
if "%CFG%"=="" goto :usage
call "%HERE%env.bat"
if errorlevel 1 goto :fail
set "OUT=results\%CFG%_ep150"
if not defined NEEDS goto :noteacher
if exist "results\%NEEDS%_ep150\models.pkl" goto :noteacher
echo [ERROR] Teacher file results\%NEEDS%_ep150\models.pkl not found. Run or copy that run first.
goto :fail
:noteacher
if not exist "%OUT%" goto :train
echo [ERROR] %OUT% already exists. A run never overwrites a folder.
echo         If the earlier attempt failed, rename it to results\_failed_%CFG% and start again.
goto :fail
:train
echo.
echo === Training %CFG% ===
python seq_NN_holdout_eval.py %COMMON% --cfg-name %CFG% --output-dir results/%CFG%_ep150
if errorlevel 1 goto :fail
python seq_NN_honest_curve.py results/%CFG%_ep150 > results/%CFG%_ep150/honest_curve.txt
if errorlevel 1 goto :fail
shift
:cmp
if "%~1"=="" goto :ok
call "%HERE%compare.bat" %CFG% %~1
shift
goto :cmp
:ok
echo.
echo === %CFG% finished. Fill in its row in RESULTS_LOG.md. ===
set "RC=0"
goto :end
:usage
echo Usage: _run.bat CFG [BASE_RUN ...]
set "RC=2"
goto :end
:fail
echo.
echo === %CFG% STOPPED. See the message above. ===
set "RC=1"
:end
rem Pause only when double-clicked. queue.bat sets NOPAUSE so unattended runs never wait for a key.
if not defined NOPAUSE (
  echo "%cmdcmdline%" | find /i "/c" >nul && pause
)
exit /b %RC%
