@echo off
rem Run 13 of COMMANDS.md: cnx_fastvit_kd_xslope_s2
setlocal EnableExtensions
set "NEEDS=x_slope_s2"
call "%~dp0_run.bat" cnx_fastvit_kd_xslope_s2 cnx_fastvit_s2 cnx_fastvit_kd_small_s2
exit /b %errorlevel%
