@echo off
REM 设置嵌入的 Python 解释器路径
set PYTHON_PATH=.\python\python.exe

REM 启动后端前端
echo [信息] 正在启动后端服务中...
"%PYTHON_PATH%" ".\lightrag_api_openai_compatible.py"
if errorlevel 1 (
    echo [错误] 运行后端失败，请查看抛出的错误并进行排查。
    pause
    exit /b
)




REM 提示完成
echo [成功] 运行成功，可开始使用！
pause