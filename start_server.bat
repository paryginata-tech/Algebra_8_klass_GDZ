@echo off
chcp 65001 >nul
cd /d "%~dp0docs"
echo Запускаю локальный сервер для решебника...
echo После запуска откройте в браузере: http://localhost:8000
echo Чтобы остановить сервер — закройте это окно.
echo.
start "" http://localhost:8000
python -m http.server 8000
pause
