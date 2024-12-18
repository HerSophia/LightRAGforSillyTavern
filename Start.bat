@echo off
REM 打开新的 CMD 窗口运行 Run API.bat
start cmd /k "call \"Run API.bat\""

REM 打开新的 CMD 窗口运行 Run Web.bat
start cmd /k "call \"Run Web.bat\""

REM 提示完成
echo 两个后端已启动：API 和 Web。

exit