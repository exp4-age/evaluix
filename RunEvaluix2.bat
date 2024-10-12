@echo off

@rem Make the console stay open when double clicking
if "%parent%"=="" set parent=%~0
if "%console_mode%"=="" (set console_mode=1& for %%x in (%cmdcmdline%) do if /i "%%~x"=="/c" set console_mode=0)

:conda-search
@rem Check if conda is installed
for /f "delims=" %%i in ('where conda 2^>nul') do set "CONDA_EXE=%%i"

@rem If conda was not found in PATH, check specific directories
if not defined CONDA_EXE (
    for %%d in (
        "%USERPROFILE%\Anaconda3\Scripts"
        "%USERPROFILE%\Miniconda3\Scripts"
    ) do (
        if exist "%%~d\conda.exe" (
            set "CONDA_EXE=%%~d\conda.exe"
            goto :activate-env
        )
    )
)

if not defined CONDA_EXE (
    @echo Conda not found.
    goto :end
)

:activate-env
@rem Get the directory of the conda executable
for %%i in ("%CONDA_EXE%") do set "CONDA_DIR=%%~dpi"

@rem Activate Evaluix2 environment
call "%CONDA_DIR%..\condabin\conda.bat" activate Evaluix2 && goto :run
if errorlevel 1 goto :end

:run
@rem Run Evaluix2
python "\\smb.uni-kassel.de\exp4_all\01_science\05_experiments\04_Steuer-und Auswertungssoftware\01_Auswertungsprogramme\12_Evaluix2\EvaluixAlpha\Evaluix2_Main.py" %*

:end
@rem Make the console stay open when double clicking
if "%console_mode%"=="1" pause
pause