@echo off
rem compare.bat TREAT_RUN BASE_RUN
rem Runs seq_NN_robust_compare.py once, for example:  compare.bat x_slope_nosynth_s1 x_slope_s1
rem Run names are the --cfg-name values, without the _ep150 suffix.
rem Exit code 1 from the script means a REJECT verdict. That is normal; the output file is still written.
setlocal EnableExtensions
if "%~2"=="" goto :usage
call "%~dp0env.bat"
if errorlevel 1 exit /b 1
set "TREAT=results\%~1_ep150"
set "BASE=results\%~2_ep150"
if not exist "%TREAT%\" goto :notreat
if not exist "%BASE%\" goto :nobase
echo.
echo === Compare: %~2 (base) vs %~1 (treat) ===
python seq_NN_robust_compare.py --base results/%~2_ep150 --treat results/%~1_ep150 --output-dir results/%~1_ep150 > results/%~1_ep150/compare_vs_%~2.txt 2>&1
echo Done. Python exit code %errorlevel% (1 means REJECT, which is normal).
echo Output: results\%~1_ep150\compare_vs_%~2.txt
exit /b 0
:usage
echo Usage: compare.bat TREAT_RUN BASE_RUN
echo Example: compare.bat x_slope_nosynth_s1 x_slope_s1
exit /b 2
:notreat
echo [ERROR] %TREAT% not found.
exit /b 1
:nobase
echo [SKIP] %BASE% is not on this machine yet.
echo        Copy that folder here, then run: compare.bat %~1 %~2
exit /b 0
