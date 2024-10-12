@echo off

@rem Make the console stay open when double clicking
if "%parent%"=="" set parent=%~0
if "%console_mode%"=="" (set console_mode=1& for %%x in (%cmdcmdline%) do if /i "%%~x"=="/c" set console_mode=0)

:variables
@rem Set the root folder of the project
set "ROOT=\\smb.uni-kassel.de\exp4_all\01_science\05_experiments\04_Steuer-und Auswertungssoftware\01_Auswertungsprogramme\12_Evaluix2\EvaluixAlpha"
set ENV=Evaluix2
set EXE=Evaluix2
@rem Optionally supply environment name as first command line parameter
if [%1] == [] goto :conda-search
set ENV=%1

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
            goto :install-preparation
        )
    )
)

@rem If conda is not found, exit the script
if not defined CONDA_EXE (
    @echo Conda not found.
    goto :end
)

:install-preparation
@rem Get the directory of the conda executable
for %%i in ("%CONDA_EXE%") do set "CONDA_DIR=%%~dpi"

@rem Activate anaconda default environment
call "%CONDA_DIR%..\condabin\activate.bat" && goto :env-check
if errorlevel 1 goto :end

:env-check
@rem Check if environment already exists and fulfills the conditions in the yaml file
for /f "tokens=1" %%i in ('conda env list ^| findstr /b /c:"%ENV% "') do (
    if "%%i"=="%ENV%" (
        call conda env export --name %ENV% > current_env.yaml
        fc /w current_env.yaml "%ROOT%\Evaluix2EnvironmentConfig.yaml" > nul
        if errorlevel 1 goto :env-install
        @echo Environment already exists and fulfills the conditions.
        goto :set-target
    )
)

:env-install
@rem Create a new environment or update the existing one
@echo Creating environment...
if not exist "%ROOT%\Evaluix2EnvironmentConfig.yaml" (
    @echo Evaluix2EnvironmentConfig.yaml not found.
    goto :end
)
@rem if the environment already exists, it will be updated using --prune to additionally remove unused packages
if exist "%CONDA_DIR%..\envs\%ENV%" (
    call conda env update --prune -f "%ROOT%\Evaluix2EnvironmentConfig.yaml" -n %ENV%
    if errorlevel 1 goto :deactivate
    @echo Environment updated.
    goto :successful
) else (
    call conda env create -f "%ROOT%\Evaluix2EnvironmentConfig.yaml" -n %ENV%
    if errorlevel 1 goto :deactivate
    @echo Environment created.
    goto :successful
)

:successful
@echo Installation successful.
goto :end

:deactivate
@rem Finishing installation
call conda deactivate

:end
@rem Also for keeping console when double clicking
if "%parent%"=="%~0" ( if "%console_mode%"=="0" pause )