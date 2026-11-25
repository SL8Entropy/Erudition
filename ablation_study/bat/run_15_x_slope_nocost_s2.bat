@echo off
rem Run 15 of COMMANDS.md: x_slope_nocost_s2
setlocal EnableExtensions
call "%~dp0_run.bat" x_slope_nocost_s2 x_slope_s2
exit /b %errorlevel%
