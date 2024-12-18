@echo off
REM ����Ƕ��� Python ������·��
set PYTHON_PATH=.\python\python.exe

REM ���з���
echo [��Ϣ] �������з�����...
"%PYTHON_PATH%" ".\lightrag_api_openai_compatible.py"
if errorlevel 1 (
    echo [����] ���з���ʧ�ܣ���鿴�׳��Ĵ��󲢽����Ų飬���߼���������ӡ�
    pause
    exit /b
)




REM ��ʾ���
echo [�ɹ�] ���гɹ����ɿ�ʼʹ�ã�
pause