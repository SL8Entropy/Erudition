@echo off
rem Run 5 of COMMANDS.md: cnx_fastvit_kd_xslope_s1
setlocal EnableExtensions
set "NEEDS=x_slope_s1"
call "%~dp0_run.bat" cnx_fastvit_kd_xslope_s1 cnx_fastvit_s1 cnx_fastvit_kd_small_s1
exit /b %errorlevel%
