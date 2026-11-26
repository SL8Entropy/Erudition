@echo off
rem Laptop D queue (RTX 5060 8 GB): both teachers and all students, seeds 1 and 2. Students need their teacher, so keep this order.
call "%~dp0queue.bat" run_01_small_s1 run_02_x_slope_s1 run_04_cnx_fastvit_kd_small_s1 run_05_cnx_fastvit_kd_xslope_s1 run_09_small_s2 run_10_x_slope_s2 run_12_cnx_fastvit_kd_small_s2 run_13_cnx_fastvit_kd_xslope_s2
