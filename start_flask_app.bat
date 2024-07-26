@echo off
cd C:\Users\sambita\webapp
call venv\Scripts\activate
waitress-serve --listen=0.0.0.0:80 app:app
