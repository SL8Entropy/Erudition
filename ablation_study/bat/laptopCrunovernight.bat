@echo off
cd /d "%~dp0"

echo Starting run_06_x_slope_nosynth_s1 at %date% %time% | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"
cmd /c "run_06_x_slope_nosynth_s1.bat" 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"
echo Finished run_06 at %date% %time% (Exit Code: %errorlevel%) | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"

echo Starting run_07_x_slope_nocost_s1 at %date% %time% | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"
cmd /c "run_07_x_slope_nocost_s1.bat" 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"
echo Finished run_07 at %date% %time% (Exit Code: %errorlevel%) | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"

echo Starting run_08_x_slope_nogeo_s1 at %date% %time% | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"
cmd /c "run_08_x_slope_nogeo_s1.bat" 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"
echo Finished run_08 at %date% %time% (Exit Code: %errorlevel%) | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"

echo All runs complete at %date% %time% | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'laptopCovernightlog.txt' -Append"
