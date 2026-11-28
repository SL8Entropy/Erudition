@echo off
rem Run 18 of COMMANDS.md: x_slope_s3 (seed 3, added 2026-11-28)
setlocal EnableExtensions
call "%~dp0_run.bat" x_slope_s3 small_s3
exit /b %errorlevel%
