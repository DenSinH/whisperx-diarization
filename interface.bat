@echo off
rem List of common virtual environment directories
set VENV_DIRS=venv env .venv .env

rem Loop through the common virtual environment directories
for %%d in (%VENV_DIRS%) do (
    if exist %%d\Scripts\activate (
        echo Found virtual environment in %%d
        call %%d\Scripts\python interface.py
        goto end
    )
)

rem If no virtual environment was found, run with system Python
echo No virtual environment found. Running with system Python...
python interface.py

:end
