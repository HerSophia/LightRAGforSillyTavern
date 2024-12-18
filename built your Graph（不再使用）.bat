@echo off
REM 设置嵌入的 Python 解释器路径
set PYTHON_PATH=.\python\python.exe

REM 构建知识图谱
echo [信息] 正在构建知识图谱...
"%PYTHON_PATH%" ".\built your Graph.py"
if errorlevel 1 (
    echo [错误] 构建知识图谱失败，请查看抛出的错误并进行排查，或者检查网络连接。
    pause
    exit /b
)

REM 提示完成
echo [成功] 构建完成，可开始运行服务！
pause