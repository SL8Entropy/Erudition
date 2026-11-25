@echo off
rem Run 16 of COMMANDS.md: x_slope_nogeo_s2
setlocal EnableExtensions
call "%~dp0_run.bat" x_slope_nogeo_s2 x_slope_s2
exit /b %errorlevel%
