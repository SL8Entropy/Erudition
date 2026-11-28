@echo off
rem Run 21 of COMMANDS.md: cnx_fastvit_kd_xslope_s3 (seed 3, added 2026-11-28)
setlocal EnableExtensions
set "NEEDS=x_slope_s3"
call "%~dp0_run.bat" cnx_fastvit_kd_xslope_s3 cnx_fastvit_s3 cnx_fastvit_kd_small_s3
exit /b %errorlevel%
