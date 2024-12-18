@echo off
REM 设置嵌入的 Python 解释器路径
set PYTHON_PATH=.\python\python.exe

set ENV_PATH=.\.env

REM 启动管理前端
echo [信息] 正在启动管理前端中...
"%PYTHON_PATH%" ".\Gradio_web.py"
if errorlevel 1 (
    echo [错误] 运行前端失败，请查看抛出的错误并进行排查。
    pause
    exit /b
)

REM 提示完成
echo [成功] 运行成功，可开始使用！
pause