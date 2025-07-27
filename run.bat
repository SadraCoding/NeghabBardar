@echo off
call venv\Scripts\activate
start cmd /k "cd api && python modelapi.py"
timeout /t 10 /nobreak >nul
start "" "flutter_interface\build\windows\x64\runner\Release\NeghabBardar.exe"
exit
