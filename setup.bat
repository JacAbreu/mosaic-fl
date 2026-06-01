@echo off
REM setup.bat - Mosaic-FL / MOSAICO-FL
REM Configura ambiente virtual e instala dependências

set VENV_DIR=.venv

echo ========================================
echo   Mosaic-FL - Setup do Ambiente
echo ========================================

REM Verifica se python está disponível
python --version >nul 2>&1
if errorlevel 1 (
    echo Erro: python nao encontrado. Instale Python >= 3.10.
    exit /b 1
)

echo [1/4] Python detectado
for /f "tokens=*" %%a in ('python --version') do echo        %%a

REM Cria venv se nao existir
if exist %VENV_DIR% (
    echo [2/4] Ambiente virtual ja existe em .\%VENV_DIR%
) else (
    echo [2/4] Criando ambiente virtual em .\%VENV_DIR%...
    python -m venv %VENV_DIR%
)

REM Ativa venv
echo [3/4] Ativando ambiente virtual...
call %VENV_DIR%\Scripts\activate.bat

REM Instala dependencias
echo [4/4] Instalando Mosaic-FL e dependencias...
python -m pip install --upgrade pip
pip install -e .

echo.
echo ========================================
echo   Setup concluido com sucesso!
echo ========================================
echo.
echo Para ativar o ambiente futuramente:
echo   %VENV_DIR%\Scripts\activate
echo.
echo Para executar os experimentos:
echo   python -m mosaicfl.experiments.runner
echo   ou
echo   python run.py
echo.
pause
