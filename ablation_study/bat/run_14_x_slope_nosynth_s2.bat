@echo off
rem Run 14 of COMMANDS.md: x_slope_nosynth_s2
setlocal EnableExtensions
call "%~dp0_run.bat" x_slope_nosynth_s2 x_slope_s2
exit /b %errorlevel%
