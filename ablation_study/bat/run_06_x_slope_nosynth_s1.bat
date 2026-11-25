@echo off
rem Run 6 of COMMANDS.md: x_slope_nosynth_s1
setlocal EnableExtensions
call "%~dp0_run.bat" x_slope_nosynth_s1 x_slope_s1
exit /b %errorlevel%
