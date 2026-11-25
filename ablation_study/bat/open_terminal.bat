@echo off
rem Opens a cmd window that is already set up (block 0 of COMMANDS.md).
rem The bat folder is on PATH there, so you can type compare.bat or run_01_small_s1.bat by name.
start "Erudition ablation_study" cmd /k call "%~dp0env.bat"
