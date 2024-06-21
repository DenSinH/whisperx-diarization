@echo off
if "%~2"=="" (
    rem If only one argument is provided, use the default python interpreter
    set "python_interpreter=python"
    set "file=%~1"
) else (
    rem If two arguments are provided, use the specified python interpreter
    set "python_interpreter=%~1"
    set "file=%~2"
)

"%python_interpreter%" ./diarize.py -a "%file%" --whisper-model large-v3 --language nl
