@echo off
cd C:\Users\sambita\webapp
call venv\Scripts\activate
echo %date% %time% > flask_app.log
waitress-serve --listen=0.0.0.0:80 app:app >> flask_app.log 2>&1
