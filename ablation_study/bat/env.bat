@echo off
rem env.bat - block 0 of COMMANDS.md. Safe to call many times.
rem Goes to ..\pipeline, activates ..\..\.venv, sets COMMON, stops the PC sleeping on AC power.
rem Layout assumed:  Erudition\.venv   Erudition\ablation_study\bat   Erudition\ablation_study\pipeline
cd /d "%~dp0..\pipeline"
if errorlevel 1 goto :nopipeline
for %%I in ("%~dp0..\..\.venv") do set "VENV=%%~fI"
if not exist "%VENV%\Scripts\activate.bat" goto :novenv
if /i not "%VIRTUAL_ENV%"=="%VENV%" call "%VENV%\Scripts\activate.bat"
set "PATH=%~dp0;%PATH%"
rem Write Python output straight away, so console logs are complete even if a run is killed.
set PYTHONUNBUFFERED=1
set COMMON=--id 0801_V2 --source-dir experiments/bilzard --epochs 150 --device cuda --offline-timm --batch-size 4 --val-batch-size 4 --grad-accum-steps 4 --num-workers 1
powercfg /change standby-timeout-ac 0 >nul 2>&1
if errorlevel 1 echo [WARN] Could not change the sleep timeout. Set it by hand or run as Administrator.
exit /b 0
:nopipeline
echo [ERROR] Folder "%~dp0..\pipeline" not found. The bat folder must sit inside ablation_study, next to pipeline.
exit /b 1
:novenv
echo [ERROR] Virtual environment not found at %VENV%
echo         Expected it at Erudition\.venv, two levels above the pipeline folder.
exit /b 1
