@echo off
REM 设置嵌入的 Python 解释器路径
set PYTHON_PATH=.\python\python.exe

REM 检查是否存在嵌入的 Python 解释器
if not exist "%PYTHON_PATH%" (
    echo [错误] 未找到嵌入的 Python 解释器，请确保已将 Python 正确解压到 LightRAG/python 目录。
    pause
    exit /b
)

REM 检查 Python 版本
"%PYTHON_PATH%" --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到有效的 Python 安装，请检查嵌入的 Python 是否完整。
    pause
    exit /b
)

REM 检查 requirements.txt 是否存在
if not exist "requirements.txt" (
    echo [错误] 未找到 requirements.txt 文件，请确认文件存在。
    pause
    exit /b
)

REM 使用 get-pip.py 安装 pip
    echo [信息] 正在使用嵌入的 Python 安装 pip...
    "%PYTHON_PATH%" "python\get-pip.py"
    if errorlevel 1 (
        echo [错误] pip 安装失败，请检查 get-pip.py 文件和网络连接。
        pause
        exit /b
    )

REM 安装项目依赖
echo [信息] 正在使用嵌入的 Python 安装项目依赖，请稍候...
"%PYTHON_PATH%" -m pip install --no-deps -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查 requirements.txt 文件或网络连接。
    pause
    exit /b
)

REM 安装开发模式包
echo [信息] 正在以可编辑模式安装当前项目...
"%PYTHON_PATH%" -m pip install -e .
if errorlevel 1 (
    echo [错误] 可编辑模式安装失败，请检查当前目录和 Python 环境。
    pause
    exit /b
)

REM 提示完成
echo [成功] 安装完成，项目已准备好运行！
pause
