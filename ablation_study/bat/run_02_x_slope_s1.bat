@echo off
rem Run 2 of COMMANDS.md: x_slope_s1
setlocal EnableExtensions
call "%~dp0_run.bat" x_slope_s1 small_s1
exit /b %errorlevel%
