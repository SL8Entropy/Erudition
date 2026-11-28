@echo off
rem Laptop D, seed-3 round (2026-11-28): both seed-3 teachers, their two students, then D1 seed 2. Students need their teacher, so keep this order.
call "%~dp0queue.bat" run_17_small_s3 run_18_x_slope_s3 run_20_cnx_fastvit_kd_small_s3 run_21_cnx_fastvit_kd_xslope_s3 run_15_x_slope_nocost_s2
