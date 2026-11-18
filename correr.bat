@echo off
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 goto fallo
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto fallo
)

echo Iniciando servidor...
"venv\Scripts\python.exe" app.py

echo.
echo El servidor se detuvo.
pause
exit /b 0

:fallo
echo.
echo No se pudo preparar el entorno. Revisa el mensaje de error.
pause
exit /b 1
