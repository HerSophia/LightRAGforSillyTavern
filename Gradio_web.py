import ast
import json
import mimetypes
import os
import shutil
import sys
import webbrowser
import zipfile
import subprocess
import asyncio
import importlib.metadata

import pkg_resources
from IPython.terminal.ipapp import frontend_flags
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from importlib.metadata import distributions
from time import sleep
from typing import List, Tuple
from datetime import datetime, time, timedelta

import gradio as gr
import httpx
from click import style
from dotenv import load_dotenv, set_key
from pathlib import Path

from fastapi import requests
from numba.typed.listobject import new_list

from playwright.sync_api import sync_playwright
from scipy.ndimage import label
from sympy import false

load_dotenv()

# 加载 .env 文件
ENV_FILE = ".env"
env_path = Path(ENV_FILE)
Port = os.getenv("API_port","")
# 配置 RAG 后端的基础 URL
RAG_API_URL = f"http://localhost:{Port}/v1"
# Constants
SUPPORTED_FILE_TYPES = ['txt','pdf','doc','ppt','csv']
FILE_BACKUP_DIR = "./backup/files"
GRAPH_BACKUP_DIR = "./backup/graph"
ENV_VARS = os.getenv("RAG_DIR","")
BUILT_YOUR_GRAPH_SCRIPT = "./build_your_graph.py"

Start_page_IsNotShow = os.getenv("start_page_show","") == 'True'

# 创建必要的备份目录
os.makedirs(FILE_BACKUP_DIR, exist_ok=True)
os.makedirs(GRAPH_BACKUP_DIR, exist_ok=True)

# 环境变量获取与更新<开始>

def get_env_variables():
    """
    读取所有环境变量并以字典形式返回
    """
    keys = [
        "RAG_DIR",
        "file_DIR",
        "API_port",
        "OPENAI_API_KEY",
        "LLM_MODEL",
        "LLM_MODEL_TOKEN_SIZE",
        "EMBEDDING_MODEL",
        "EMBEDDING_MAX_TOKEN_SIZE",
        "OPENAI_BASE_URL",
        "start_page_IsNotShow",
        "FRONTEND_PORT",
    ]
    return {key: os.getenv(key, "") for key in keys}

def update_env_variable(key, value):
    """
    更新 .env 文件中的某个环境变量
    """
    if key not in get_env_variables():
        return f"Error: {key} is not a valid environment variable."

    set_key(env_path, key, value)
    os.environ[key] = value  # 同时更新当前环境变量
    return f"Successfully updated {key} to {value}."

def reset_env_variable(key):
    """
    重置某个环境变量为空
    """
    return update_env_variable(key, "")

# 环境变量获取与更新<结束>


# 文档文件管理<开始>

def list_files_in_folder(folder="./files"):
    """
    列出指定目录及子目录下的支持的文件类型，并返回相对路径列表，按文件名排序。

    参数:
        folder (str): 指定的根目录，默认为 "./text"。

    返回:
        list: 包含相对路径的文件列表。
    """
    all_files = []
    folder = os.path.abspath(folder)  # 获取根目录的绝对路径

    for root, _, files in os.walk(folder):
        for file in files:
            if file.split(".")[-1].lower() in SUPPORTED_FILE_TYPES:
                # 拼接文件路径并确保是相对路径
                full_path = os.path.join(root, file)
                rel_path = "." + os.path.relpath(full_path, start=os.getcwd())  # 转为相对路径
                rel_path = rel_path.replace("\\", "/")
                all_files.append(full_path)
    return sorted(all_files)

def refresh_file_list_display():
    """
    刷新文件列表，返回文件名 Markdown 列表和文件路径字典
    """
    files = list_files_in_folder()
    file_dict = {}

    for file_path in files:
        file_name = os.path.basename(file_path)

        # 如果文件名已存在，添加文件创建时间作为后缀
        if file_name in file_dict:
            creation_time = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d_%H-%M-%S')
            unique_file_name = f"{file_name}--{creation_time}"
            file_dict[unique_file_name] = file_path # 特殊文件名 -> 文件路径
        else:
            file_dict[file_name] = file_path  # 文件名 -> 文件路径
    markdown_list = "\n".join(file_dict.keys())  # 生成文件名列表
    return markdown_list, file_dict

def refresh_dropdown_choices(file_dict):
    """
    根据文件字典生成 Dropdown 的可选项
    """
    if file_dict is None:
        return []  # 防止错误发生，返回空列表
    #print(list(file_dict.keys()))
    return list(file_dict.keys())  # 返回所有文件名

def derefresh_dropdown_choices_temp():
    """
    根据文件字典生成 Dropdown 的可选项
    """
    markdown_list, file_dict = refresh_file_list_display()
    if file_dict is None:
        return []  # 防止错误发生，返回空列表
    #print(list(file_dict.keys()))
    return list(file_dict.keys())  # 返回所有文件名

def handle_file_selection(file_name, file_dict):
    """根据选择的文件名返回完整路径"""
    return file_dict.get(file_name, None)

def open_text_folder(folder_paths):
    """在文件资源管理器中打开指定文件夹，删除路径中的文件名"""
    folder_paths = eval(folder_paths)
    if isinstance(folder_paths, list):

        for folder_path in folder_paths:
            folder_path = os.path.dirname(folder_path)  # 获取文件夹路径
            try:
                if os.name == "nt":  # Windows
                    os.startfile(folder_path)
                elif os.name == "posix":  # macOS/Linux
                    os.system(f"open {folder_path}" if sys.platform == "darwin" else f"xdg-open {folder_path}")
            except Exception as e:
                return f"打开文件夹失败：{str(e)}"
    else:
        return "错误：未传入文件夹路径列表。"

    return f"已成功打开 {len(folder_paths)} 个文件夹。"

def open_text_file(file_paths):
    """使用系统默认程序打开多个文件"""
    file_paths = eval(file_paths)
    if not isinstance(file_paths, list) or len(file_paths) == 0:
        return "错误：未传入文件路径列表。"

    results = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            results.append(f"❌ 文件不存在：{file_path}")
            continue

        try:
            if os.name == "nt":  # Windows
                os.startfile(file_path)
            elif os.name == "posix":  # macOS/Linux
                os.system(f"open {file_path}" if sys.platform == "darwin" else f"xdg-open {file_path}")
            results.append(f"✅ 文件 {file_path} 已打开。")
        except Exception as e:
            results.append(f"❌ 打开文件失败：{file_path}，错误：{str(e)}")

    return "\n".join(results)

def set_rag_env_variable(file_paths):
    """
    设置 file_DIR 环境变量的值为指定文件的路径。

    参数:
    - file_path (str): 文件路径，必须是相对路径且在 ./text/ 目录下。

    返回:
    - str: 设置结果信息。
    """
    file_paths = eval(file_paths)
    file_path = ("./" + os.path.relpath(file_paths[0], start=os.getcwd())).replace("\\", "/")  # 转为相对路径
    print(file_path)
    # 验证文件路径是否符合要求
    if not file_path.startswith("./files/"):
        return "Error: 文件路径必须位于 ./files/ 目录下。"

    if not os.path.isfile(file_path):
        return f"Error: 文件 {file_path} 不存在。"

    # 获取当前 file_DIR 的值
    current_value = os.getenv("file_DIR", "")
    reset_result = reset_env_variable("file_DIR")  # 重置 RAG_DIR 环境变量

    if "Error" in reset_result:
        return f"重置 file_DIR 失败: {reset_result}"

    # 将路径转换为 Windows 风格（用反斜杠）
    windows_style_path = file_path.replace("\\", "/")

    # 更新 .env 文件和环境变量
    update_result = update_env_variable("file_DIR", windows_style_path)

    if "Successfully updated" in update_result:
        return (
            f"file_DIR 更新成功！\n"
            f"旧值: {current_value}\n"
            f"新值: {windows_style_path}"
        )
    else:
        return f"更新失败: {update_result}"

def delete_file_with_backup(file_paths):
    """删除多个文件，先备份后删除"""
    file_paths = eval(file_paths)
    if not isinstance(file_paths, list) or len(file_paths) == 0:
        return "错误：未传入文件路径列表。"

    results = []
    for file_path in file_paths:
        try:
            if not os.path.isfile(file_path):
                results.append(f"❌ 文件不存在：{file_path}")
                continue

            backup_name = os.path.join(
                FILE_BACKUP_DIR,
                f"{os.path.basename(file_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(file_path, backup_name)  # 备份文件
            os.remove(file_path)  # 删除文件
            results.append(f"✅ 文件 {file_path} 已删除，备份存储在 {backup_name}")
        except Exception as e:
            results.append(f"❌ 删除失败：{file_path}，错误：{str(e)}")

    return "\n".join(results)

def create_unique_folder(file_name):
    """
    根据文件名在 ./files 中创建唯一的文件夹，不包含文件格式后缀。
    """
    base_folder = "./files"
    # 去除文件名的后缀
    folder_name = os.path.splitext(file_name)[0]
    folder_path = os.path.join(base_folder, folder_name)

    # 如果文件夹已存在，则生成带有时间戳的唯一文件夹名称
    if os.path.exists(folder_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{folder_name}_{timestamp}"
        folder_path = os.path.join(base_folder, folder_name)

    # 创建文件夹
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def upload_files_and_save(files):
    """
    上传多个文件并保存到新创建的文件夹中。
    :param files: 上传的文件列表（Gradio 返回的文件列表）
    :return: 每个文件的处理结果列表
    """
    if not files or len(files) == 0:
        return "未上传任何文件。"

    results = []  # 用于存储每个文件的处理结果
    uploaded_files = {}  # 用于存储成功上传文件的字典

    for file in files:
        file_name = os.path.basename(file)
        try:
            # 确定文件名和扩展名
            file_ext = file_name.split('.')[-1].lower()

            # 验证文件类型
            if file_ext not in SUPPORTED_FILE_TYPES:
                results.append(f"❌ 不支持的文件类型: {file_name} ({file_ext})。支持的类型包括: {', '.join(SUPPORTED_FILE_TYPES)}")
                continue

            # 创建唯一文件夹
            folder_path = create_unique_folder(file_name)
            os.makedirs(folder_path, exist_ok=True)

            # 目标文件路径
            file_path = os.path.join(folder_path, file_name)

            # 获取 Gradio 返回的文件路径并复制到目标路径
            shutil.copy(file, file_path)

            # 记录成功上传的文件
            uploaded_files[file_name] = file_path

            results.append(f"✅ 文件 {file_name} 上传成功，已保存至文件夹: {folder_path}")
        except Exception as e:
            results.append(f"❌ 文件 {file_name} 上传过程中出现错误: {str(e)}")

    return "\n".join(results),uploaded_files,uploaded_files

def debug_file(file):
    if not file:
        return "未上传任何文件。"

    try:
        return {
            "文件名": file.name,
            "文件类型": str(type(file)),
            "支持的操作": dir(file),
        }
    except Exception as e:
        return f"调试文件信息时出错: {str(e)}"

def build_graph_for_files(prebuild_dict:dict):
    """
    构建知识图谱：调用服务端接口处理多个文件
    :param prebuild_dict: 字典，key 是文件名，value 是文件路径
    :return: 多文件的构建结果
    """
    if isinstance(prebuild_dict, str):
        prebuild_dict = ast.literal_eval(prebuild_dict)
    base_path = "./graph"
    file_name = list(prebuild_dict.keys())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    graph_path = os.path.join(base_path, file_name[0])

    if os.path.exists(graph_path):
        graph_path = f"{graph_path}_{timestamp}"
    os.makedirs(graph_path, exist_ok=True)
    graph_path = graph_path.replace("\\", "/")
    update_env_variable("RAG_DIR", graph_path)
    try:
        # 调用文件上传与处理接口
        sleep(1.0)
        response = asyncio.run(upload_files_to_rag(prebuild_dict))

        # 如果响应中包含文件处理结果，则返回
        if isinstance(response, list):
            return response  # 返回服务端的文件处理结果列表
        else:
            # 否则返回错误信息
            return [{"status": "failed", "message": response.get("message", "Unknown error")}]
    except Exception as e:
        return [{"status": "failed", "message": f"Failed to build graph: {str(e)}"}]

def insert_graph_for_files(preinsert_dict):
    """
    构建知识图谱：调用服务端接口处理多个文件
    :param preinsert_dict: 字典，key 是文件名，value 是文件路径
    :return: 多文件的插入结果
    """
    if isinstance(preinsert_dict, str):
        preinsert_dict = ast.literal_eval(preinsert_dict)
    try:
        # 调用文件上传与处理接口
        response = asyncio.run(upload_files_to_rag(preinsert_dict))

        # 如果响应中包含文件处理结果，则返回
        if isinstance(response, list):
            return response  # 返回服务端的文件处理结果列表
        else:
            # 否则返回错误信息
            return [{"status": "failed", "message": response.get("message", "Unknown error")}]
    except Exception as e:
        return [{"status": "failed", "message": f"Failed to insert graph: {str(e)}"}]


async def upload_files_to_rag(prebuild_dict_result, purpose="knowledge_graph_frontend"):
    """
    上传文件名和路径字典到 RAG 系统
    :param prebuild_dict_result: 字典，key 是文件名，value 是文件路径
    :param purpose: 上传的目的
    :return: 服务端返回的多文件处理结果
    """
    load_dotenv(override=True)
    retries = 5  # 最大重试次数
    async with httpx.AsyncClient(timeout=300.0) as client:
        for attempt in range(1, retries + 1):
            try:
                # 构造请求体
                payload = {
                    "files": prebuild_dict_result,
                    "purpose": purpose,
                }
                response = await client.post(RAG_API_URL + f"/files", json=payload)

                # 检查响应状态
                if response.status_code == 200:
                    find_html_file(os.getenv("RAG_DIR"))
                    return response.json()  # 成功返回 JSON 数据
                else:
                    find_html_file(os.getenv("RAG_DIR"))
                    return {
                        "status": "failed",
                        "message": f"Server returned error: {response.status_code}, {response.text}",
                    }
            except Exception as e:
                if attempt == retries:
                    return {
                        "status": "failed",
                        "message": f"Failed to communicate with server: {str(e)}",
                    }

def debug_and_return(name):
    """
    返回文件名，同时输出调试信息
    """
    #debug_message = f"调试：当前选择的文件名是 {name}"
    #print(debug_message)  # 控制台调试
    return name


# 文档文件管理<结束>


# 图谱管理<开始>
'''
def setup_file_upload_interaction(file_uploader, purpose_input, upload_button, upload_result):
    """设置文件上传交互逻辑"""
    upload_button.click(
        fn=upload_file_to_rag,
        inputs=[file_uploader, purpose_input],
        outputs=upload_result,
    )
'''
def list_subdirectories(base_path="./graph"):
    """
    列出指定文件夹下的所有次级文件夹，并返回文件夹名称与其绝对路径的映射字典。
    对于重名文件夹，添加创建时间后缀以区分。
    """
    if not os.path.exists(base_path):
        return {}, "The specified base path does not exist."

    subdirectories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    folder_dict = {}

    for folder in subdirectories:
        folder_path = os.path.join(base_path, folder)
        if folder in folder_dict:
            # 如果出现同名文件夹，添加创建时间后缀
            creation_time = datetime.fromtimestamp(os.path.getctime(folder_path)).strftime('%Y-%m-%d_%H-%M-%S')
            unique_folder_name = f"{folder}--{creation_time}"
            folder_dict[unique_folder_name] = os.path.abspath(folder_path)
        else:
            folder_dict[folder] = os.path.abspath(folder_path)
    # 生成 Markdown 格式文件名列表
    markdown_list = "\n".join(list(folder_dict.keys()))
    selective_list = list(folder_dict.keys())
    return markdown_list,folder_dict,selective_list

def open_rag_folder(folder_path):
    """在文件资源管理器中打开指定文件夹"""
    if os.name == "nt":  # Windows
        os.startfile(folder_path)
    elif os.name == "posix":  # macOS/Linux
        os.system(f"open {folder_path}" if sys.platform == "darwin" else f"xdg-open {folder_path}")

def backup_and_delete_graph_folder(selected_graph_abs_path):
    """
    备份知识图谱文件夹并删除原路径
    :param selected_graph_abs_path: 即将要删除的路径（绝对路径）
    :return: 操作结果字符串
    """
    try:
        # 检查路径有效性
        if not selected_graph_abs_path or not os.path.exists(selected_graph_abs_path):
            return "无法备份，路径不存在或未提供。"

        # 提取文件夹名作为变量
        folder_name = os.path.basename(os.path.normpath(selected_graph_abs_path))
        PreBackup_folder = f"{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(GRAPH_BACKUP_DIR, PreBackup_folder)

        # 创建备份文件夹
        os.makedirs(GRAPH_BACKUP_DIR, exist_ok=True)

        # 备份路径
        shutil.copytree(selected_graph_abs_path, backup_path)

        # 删除原路径
        shutil.rmtree(selected_graph_abs_path)

        return f"备份成功！图谱已备份至 {backup_path}，并成功删除图谱。"
    except Exception as e:
        return f"备份或删除过程中出现错误: {str(e)}"

def find_html_file(folder_path, filename="knowledge_graph.html"):
    """在指定文件夹下递归查找 HTML 文件"""
    for root, _, files in os.walk(folder_path):
        if filename in files:
            file_path = os.path.join(root, filename)
            webbrowser.open(file_path)
            return os.path.join(root, filename)
    return None

# 单个 ZIP 文件解压逻辑
async def upload_and_extract_zip(file, base_path="./files"):
    """上传并解压 zip 文件"""
    folder_name = os.path.splitext(os.path.basename(file.name))[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dest_folder = os.path.join(base_path, folder_name)

    if os.path.exists(dest_folder):
        dest_folder = f"{dest_folder}_{timestamp}"
    os.makedirs(dest_folder, exist_ok=True)

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            # 尝试以 GBK 解码
            for zip_info in zip_ref.infolist():
                zip_info.filename = zip_info.filename.encode('cp437').decode('utf-8')  # 转换编码
                zip_ref.extract(zip_info, dest_folder)
            print("utf8")
    except UnicodeDecodeError:
        # 使用 GBK 重读 ZIP 文件
        with zipfile.ZipFile(file, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                zip_info.filename = zip_info.filename.encode('cp437').decode('gbk')  # 转换编码
                zip_ref.extract(zip_info, dest_folder)
            print("gbk")


    return f"✅ 文件 {file.name} 已解压至: {dest_folder}"

# 将多个文件处理的逻辑拆分出来
async def process_uploaded_zips_with_progress(files,progress=gr.Progress(track_tqdm=True)):
    """处理多个 ZIP 文件的解压逻辑，使用 Gradio 进度条"""
    if not files or len(files) == 0:
        return "未上传任何文件。"
    idx = 0
    results = []
    total_files = len(files)
    progress(0,desc="正在处理中，请稍后...",total=total_files)# 使用 Gradio 的进度条
    for file in progress.tqdm(files):
        try:
            result = await upload_and_extract_zip(file)
            results.append(result)
        except Exception as e:
            results.append(f"❌ 文件 {file.name} 解压失败: {str(e)}")
        progress.update(idx + 1)  # 更新进度

    return "\n".join(results)


def set_env_variable_from_folder(folder_path):
    """将文件夹路径设置为环境变量"""
    update_env_variable("RAG_DIR", folder_path)
    return f"已将路径 {folder_path} 设置为环境变量 RAG_DIR"

# 图谱管理<结束>

# 欢迎界面<开始>

def load_readme():
    """加载 README.md 内容"""
    try:
        with open("README.md", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "README.md 文件未找到，请检查项目目录。"

def load_license():
    """加载开源协议内容"""
    try:
        with open("LICENSE", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "LICENSE 文件未找到，请检查项目目录。"


def load_requirements(file_path="requirements.txt"):
    """
    读取 requirements.txt 中的依赖包信息，并支持复杂版本约束。
    :param file_path: requirements.txt 文件路径
    :return: (包名列表, 完整依赖行列表)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        package_names = []
        valid_requirements = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                # 忽略空行和注释行
                continue

            try:
                # 解析依赖项
                req = Requirement(line)
                package_names.append(req.name.lower())
                valid_requirements.append(line)
            except Exception as e:
                # 如果某行不是有效的依赖格式，输出警告或跳过
                print(f"⚠️ 无法解析的依赖项: {line}. 错误: {e}")

        return package_names, valid_requirements

    except FileNotFoundError:
        raise FileNotFoundError(f"找不到指定的文件: {file_path}")
    except Exception as e:
        raise ValueError(f"加载依赖项时出错: {str(e)}")

def check_installed_packages():
    """
    获取当前环境中已安装的依赖包及版本。
    """
    installed_packages = {
        dist.key: dist.version for dist in pkg_resources.working_set
    }
    return installed_packages

def parse_requirement(requirement):
    """
    解析依赖项（支持复杂版本约束）。
    """
    try:
        req = Requirement(requirement)
        return req.name.lower(), req.specifier
    except Exception as e:
        raise ValueError(f"无法解析依赖项: {requirement}. 错误: {str(e)}")

def check_dependency_status():
    """检查依赖包状态"""
    required_packages, full_requirements = load_requirements()
    installed_packages = check_installed_packages()

    missing_packages = []
    mismatched_versions = []

    for req in full_requirements:
        try:
            pkg, specifier = parse_requirement(req)
            if pkg not in installed_packages:
                missing_packages.append(f"🚫 {req}")
            else:
                installed_version = Version(installed_packages[pkg])
                if not specifier.contains(installed_version):
                    mismatched_versions.append(
                        f"⚠️ {pkg} (expected {specifier}, found {installed_version})"
                    )
        except InvalidVersion as e:
            mismatched_versions.append(f"⚠️ 无法解析版本: {req}. 错误: {str(e)}")

    if not missing_packages and not mismatched_versions:
        return "✅ 所有依赖包已安装", [], []
    else:
        return (
            "部分依赖包存在问题，请查看下方列表。",
            missing_packages,
            mismatched_versions,
        )


def install_missing_packages(missing_packages):
    """安装缺失的依赖包"""
    try:
        for package in missing_packages:
            pkg = package.split(" ")[1]  # 提取包名（忽略符号 🚫）
            subprocess.check_call(["pip", "install", pkg])
        return "✅ 缺失的依赖包已成功安装"
    except subprocess.CalledProcessError as e:
        return f"❌ 安装失败: {e}"


# 安装按钮逻辑
def install_and_update(missing_packages):
            if not missing_packages:
                return "没有需要安装的依赖包"
            install_result = install_missing_packages(missing_packages)
            status, _, _ = check_dependency_status()  # 检查安装后的状态
            return status, install_result

async def check_lightrag_status():
    """检查 LightRAG 后端状态"""
    retries = 5  # 最大重试次数
    async with httpx.AsyncClient(timeout=5.0) as client:  # 设置超时时间
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(RAG_API_URL + "/connect")
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and data.get("connective") is True:
                        return "✅LightRAG 后端运行正常"
            except (httpx.ConnectError, httpx.TimeoutException):
                # 捕获连接错误或超时
                if attempt == retries:
                    return "❌LightRAG 后端未正常运行"
                continue  # 继续重试
    return "❌LightRAG 后端未正常运行，可点击💻以尝试启动"

def check_model_connection_status():
    """检查大模型连接状态"""
    # 示例实现，可以扩展为实际模型连接的检查逻辑
    return "✅大模型连接成功"

def check_port():
    port = os.getenv("API_port","")
    web = f"http://localhost:{port}/v1"
    return web

# 刷新按钮逻辑
def refresh_status():
    status, missing, mismatched = check_dependency_status()
    return (
        status,
        missing + mismatched,  # 展示所有缺失和版本问题
        bool(missing or mismatched),
    )

# 欢迎界面<结束>


# HTML to Graph<开始>

# 全局状态
STATE = {
    "notification_hidden_until": "2024-12-5 00:00:00",  # 通知栏隐藏截止时间
    "dependencies_installed": False    # 是否已安装依赖
}

# 检查依赖是否已安装
def check_dependencies():
    # 假设依赖为某个 pip 包，例如 'some_package'
    try:
        import some_package
        return True
    except ImportError:
        return False

# 安装依赖逻辑
def install_dependencies():
    try:
        os.system("pip install some_package")  # 替换为实际依赖
        return True, "依赖安装成功！"
    except Exception as e:
        return False, f"安装依赖时出错: {str(e)}"

def handle_notification_action(action, remember=False):
    today = datetime.today()  # 当前日期时间
    if action == "install":
        success, message = install_dependencies()
        STATE["dependencies_installed"] = success
        return message, success
    elif action == "dismiss":
        if remember:
            STATE["notification_hidden_until"] = str(datetime.strptime(str(today + timedelta(days=7)),"%Y-%m-%d"))
        else:
            STATE["notification_hidden_until"] = "2024-12-1 00:00:00"
        return True, True

# 构建通知栏逻辑
def should_show_notification():
        """判断是否应该显示通知栏"""
        today = datetime.today()
        hidden_until = datetime.strptime(STATE.get("notification_hidden_until"), "%Y-%m-%d %H:%M:%S")
        diff = today - hidden_until
        #print(diff.days >= 7)
        return diff.days >= 7 # 如果超过7天，则显示通知栏

def handle_install_dependencies():
        """处理安装依赖的逻辑"""
        STATE["dependencies_installed"], message = install_dependencies()
        return message, should_show_notification()

def close_notification(remember):
        """关闭通知栏逻辑"""
        if remember:
            STATE["notification_hidden_until"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        return False, False,True

# 构建通知栏
def notification_ui():
    with gr.Blocks() as notification_ui:
        notification_bar = gr.Group()  # 通知栏

        with notification_bar:
            gr.Markdown("### 通知：此页面为可选功能，依赖尚未安装。")
            gr.Markdown("请根据需求安装依赖，或直接开始使用。")
            install_btn = gr.Button("我已知晓并开始安装相关依赖")
            close_btn = gr.Button("开始使用")
            remember_checkbox = gr.Checkbox(label="七天内不再显示")

            # 按钮交互
            install_btn.click(
                fn=handle_install_dependencies,
                inputs=[],
                outputs=[notification_bar, notification_ui]
            )
            close_btn.click(
                fn=close_notification,
                inputs=[remember_checkbox],
                outputs=[notification_bar, notification_ui]
            )

    return notification_ui,"调试：通知栏"

# 转换HTML到PDF函数
def html_to_pdf(urls, output_dir="./PDF_generate"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generated_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for url in urls:
            page = browser.new_page()
            page.goto(url)
            domain = url.split("//")[-1].split("/")[0]  # 提取域名
            file_name = f"{domain}-{timestamp}.pdf"
            file_dir = os.path.join(output_dir, domain)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            output_path = os.path.join(file_dir, file_name)
            page.pdf(path=output_path)
            generated_files.append(output_path)
        browser.close()
    return generated_files

# 打开 PDF 功能
def open_pdf(filepath):
    if os.path.exists(filepath):
        os.system(f"start {filepath}")  # Windows 上打开文件
        return f"打开 PDF 文件: {filepath}"
    else:
        return "文件不存在，请检查路径！"

# 删除PDF并备份
def delete_pdf_with_backup(pdf_paths, backup_dir="./backup/PDF_generate"):
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    deleted_files = []
    for pdf in pdf_paths:
        if os.path.exists(pdf):
            backup_path = os.path.join(backup_dir, os.path.basename(pdf))
            shutil.move(pdf, backup_path)
            deleted_files.append((pdf, backup_path))
    return deleted_files

# HTML to Graph<结束>

# 侧边栏<开始>

# 通过后端服务获取模型信息
async def fetch_model_info(base_url, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(RAG_API_URL+ f"/models")
            response.raise_for_status()
            response_data = response.json()
            all_models = [model["id"] for model in response_data.get("data", [])]
            # 筛选逻辑
            large_models = [
                m for m in all_models
                if isinstance(m, str) and "embedding" not in m.lower() and "embed" not in m.lower()
            ]
            embed_models = [
                m for m in all_models
                if isinstance(m, str) and ("embedding" in m.lower() or "embed" in m.lower())
            ]
            return large_models, embed_models
        except Exception as e:
            return str(e), []

# 获取环境变量的初始值
def get_initial_values():
    base_url = os.getenv("OPENAI_BASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    API_port = os.getenv("API_port", "")
    LLM = os.getenv("EMBEDDING_MODEL","")
    EMBED = os.getenv("EMBEDDING_MODEL","")
    api_key_display = "API_KEY已保存" if api_key else ""
    return base_url, api_key_display,API_port

# 检查并补全 BASE_URL
def normalize_base_url(base_url):
    base_url = base_url.strip()  # 去除首尾空格
    if not base_url.endswith("/v1"):  # 检查是否以 /v1 结尾
        if not base_url.endswith("/"):  # 如果没有末尾的斜杠，先添加
            base_url += "/"
        base_url += "v1"
    return base_url

def load_model_configs(json_file):
    """
    从 JSON 文件中动态加载模型配置，并按照模型类别存入字典。
    :param json_file: JSON 文件路径
    :return: (LLM 模型字典, Embedding 模型字典)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        llm_models = {}
        embedding_models = {}

        # 遍历 JSON 数据并分类
        for model_category, models in data.items():
            if model_category == "LLM":
                llm_models.update(models)
            elif model_category == "Embedding":
                embedding_models.update(models)

        return llm_models, embedding_models
    except Exception as e:
        raise ValueError(f"加载 JSON 文件失败: {str(e)}")

def get_max_tokens(llm_name, embedding_name):
    """
    根据模型名称获取其对应的 Max_tokens
    :param llm_name: 大模型名称
    :param embedding_name: 嵌入模型名称
    :return: (LLM 模型 Max_tokens, Embedding 模型 Max_tokens)
    """
    llm_dict,embedding_dict = load_model_configs("./models.json")
    llm_tokens = llm_dict.get(llm_name, None)
    embedding_tokens = embedding_dict.get(embedding_name, None)

    if llm_tokens is None:
        raise ValueError(f"大模型 '{llm_name}' 的 Max_tokens 未找到。")
    if embedding_tokens is None:
        raise ValueError(f"嵌入模型 '{embedding_name}' 的 Max_tokens 未找到。")

    return llm_tokens, embedding_tokens

# 保存设置的逻辑
def save_settings(base_url, api_key,port,llm_max_tokens,embed_max_tokens):
    base_url = normalize_base_url(base_url)  # 检查并补全 BASE_URL
    api_key = api_key.strip()
    port = str(port).strip()
    llm_max_token = str(llm_max_tokens).strip()
    embed_max_token = str(embed_max_tokens).strip()
    Port = os.getenv("API_port","") #全局变量更新
    update_env_variable("OPENAI_BASE_URL", base_url)
    update_env_variable("API_port",port)
    update_env_variable("LLM_MODEL_TOKEN_SIZE",llm_max_token)
    update_env_variable("EMBEDDING_MAX_TOKEN_SIZE",embed_max_token)
    if api_key and api_key != "API_KEY已保存":
        os.environ["OPENAI_API_KEY"] = api_key

# 侧边栏<结束>

# UI 构建模块化函数

def notification_bar():
    """
    纯 CSS 实现右上角通知栏，支持多条消息堆叠和自动消失。还暂时不可用。
    """
    html_content = """
    <style>
    #notifications-container {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000; /* 确保通知栏显示在最上层 */
        display: flex;
        flex-direction: column;
        gap: 10px; /* 通知栏之间的间距 */
        pointer-events: none; /* 确保鼠标点击穿透到主页面 */
    }

    .notification {
        background-color: #4caf50; /* 默认绿色通知 */
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        opacity: 0; /* 初始透明 */
        transform: translateY(-20px); /* 初始上移 */
        animation: slideInOut 5s ease-in-out forwards; /* 动画控制显示与隐藏 */
    }

    .notification.error {
        background-color: #f44336; /* 红色通知 */
    }

    .notification.warning {
        background-color: #ff9800; /* 橙色通知 */
    }

    .notification.success {
        background-color: #4caf50; /* 绿色通知 */
    }

    @keyframes slideInOut {
        0% { opacity: 0; transform: translateY(-20px); }
        10% { opacity: 1; transform: translateY(0); }
        90% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-20px); }
    }
    </style>
    <div id="notifications-container"></div>
    <script>
    function addNotification(message, type = 'success') {
        const container = document.getElementById('notifications-container');
        if (!container) return;

        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        // 添加到容器中
        container.appendChild(notification);

        // 自动移除通知
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.addEventListener('transitionend', () => notification.remove());
        }, 5000); // 5 秒后自动删除
    }

    // 测试用通知（可移除）
    setTimeout(() => addNotification('保存成功！', 'success'), 1000);
    setTimeout(() => addNotification('保存失败：请检查输入！', 'error'), 2000);
    setTimeout(() => addNotification('警告：API Key 将过期！', 'warning'), 3000);
    </script>
    """
    return html_content

def sidebar_ui():
    custom_css = """
        .SideBar {
            width: auto !important;
            height: 100% !important;
            max-width: 25% !important;
            background-color: #f5f5f5;
            padding: 10px;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
        }

        .Closed-SideBar {
            width: auto !important;
            height: 100% !important;
            max-width: 5% !important;
            background-color: #f5f5f5;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: flex-end; /* 将内容靠右对齐 */
        }
        
        #Closed-SideBar-button {
            width: auto !important;
            height: 100% !important;
            max-width: 5% !important;
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.3s ease-in-out;
            display: flex;
            justify-content: flex-end; /* 将内容靠右对齐 */
        }

        .gradient-button {
            background: linear-gradient(90deg, #4caf50, #8bc34a);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.3s ease-in-out;
        }

        .gradient-button:hover {
            background: linear-gradient(90deg, #8bc34a, #4caf50);
        }
        
        #ASideBar {
            text-align: center; /* 居中对齐 */
            font-size: 28px; /* 字体大小 */
            font-weight: bold; /* 加粗 */
            background-color: #f5f5f5; /* 背景色与侧边栏一致 */
            padding: 15px; /* 内边距 */
            border-radius: 5px; /* 圆角 */
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* 阴影效果 */
            margin-bottom: 20px; /* 下边距 */
            color: #333; /* 字体颜色 */
        }
    """
    #gr.HTML(notification_bar())
    with gr.Blocks() as Thesidebar:
        with gr.Column(elem_classes="SideBar") as SideBar:
            with gr.Row():
                #gr.Markdown("侧边栏",elem_id="ASideBar")
                close_button = gr.Button("❌ 关闭侧边栏", elem_id="close-sidebar", elem_classes="gradient-button")

            # 输入 BASE_URL 和 API_KEY
            base_url_input = gr.Textbox(label="BASE_URL", placeholder="请输入 BASE_URL(注意要添加/v1)")
            api_key_input = gr.Textbox(label="API_KEY", placeholder="请输入 API_KEY")
            API_port_input = gr.Textbox(label="API_PORT",placeholder="请填入你想设置的RAG系统端口")


            # 大模型和嵌入模型下拉框
            large_model_dropdown = gr.Dropdown(label="选择大模型", elem_id="llms",choices=[],interactive=True)
            llm_MAX_tokens = gr.Textbox(label="大模型的Max_tokens",elem_id="llm_max_tokens",placeholder="请查询所使用的大模型的Max_tokens并填入")
            embed_model_dropdown = gr.Dropdown(label="选择嵌入模型", elem_id="embedding",choices=[],interactive=True)
            embed_MAX_tokens = gr.Textbox(label="嵌入的Max_tokens",elem_id="embed_max_tokens",placeholder="请查询所使用的嵌入模型的Max_tokens并填入")

            gr.Markdown("""
                ### ℹ️ Tips：
                    - 以上为主要选项。
                    - 请记得填入基本信息然后保存。
                    - 请刷新模型信息，选择模型或者填入Tokens后会自动保存对应信息。
                    - 如果Max_tokens遇到了错误，请在根目录的models.json中添加你使用的模型以及对应的tokens。
                    """,
                    show_copy_button=False,
                    container=True
            )
            # 保存设置按钮
            save_button = gr.Button("保存", elem_id="save-settings", elem_classes="gradient-button")

            # 获取模型信息按钮
            fetch_models_button = gr.Button("刷新模型信息", elem_id="fetch-models", elem_classes="gradient-button")

            # 选择上下文策略
            with gr.Row():
                gr.Dropdown(
                    label="选择上下文策略（暂时不可用）",
                    choices=["策略1", "策略2", "策略3"],
                    value="策略1",
                    interactive=True
                )
                gr.Dropdown(
                    label="选择 Prompt（暂时不可用）",
                    choices=["Prompt1", "Prompt2", "Prompt3"],
                    value="Prompt1",
                    interactive=True
                )


            with gr.Accordion(label="次要设置（暂时不可用）",elem_id="Addition") as addition:
                with gr.Column():
                    gr.Textbox(label="次要BASE_URL",elem_id="sub_BASE_URL",placeholder="请输入次要BASE_URL(注意要添加/v1)")
                    gr.Textbox(label="次要API_KEY", elem_id="sub_API_KEY",placeholder="请输入次要API_KEY")

                    # 大模型和嵌入模型下拉框
                    sub_large_model_dropdown = gr.Dropdown(label="选择次要大模型", elem_id="sub_llms", choices=[], interactive=True)
                    gr.Markdown("""
                        ### ℹ️ Tips：
                        - 以上为次要选项。
                        - 在有Rate Limit的情况下可以选择使用，RAG系统会使用以上设定以执行不那么重要的任务。
                        - 例如问答或者聊天时提取关键词会使用次要大模型，而不是主要的大模型。
                        """)

            # 交互逻辑
            save_button.click(
                fn=save_settings,
                inputs=[base_url_input, api_key_input,API_port_input],
                outputs=None
            )
            save_button.click(
                lambda url, key,port,llm_max_tokens,embed_max_tokens: f"<script>addNotification('保存成功！', 'success');</script>"
                if url and key and port and llm_max_tokens and embed_max_tokens else
                f"<script>addNotification('保存失败：请输入完整信息！', 'error');</script>",
                inputs=[base_url_input, api_key_input,API_port_input,llm_MAX_tokens,embed_MAX_tokens],
                outputs=None
            )

            def return_model_info(base_url_input, api_key_input):
                large_models, embed_models = asyncio.run(fetch_model_info(base_url_input, api_key_input))
                update_env_variable("LLM_MODEL", large_models[0])
                update_env_variable("EMBEDDING_MODEL", embed_models[0])
                return gr.update(elem_id="llms",choices=large_models,value=large_models[0]),gr.update(elem_id="embedding",choices=embed_models,value=embed_models[0])

            fetch_models_button.click(
                fn=return_model_info,
                inputs=[base_url_input, api_key_input],
                outputs=[large_model_dropdown, embed_model_dropdown],
            )
            def large_model_dropdown_input(large_model_dropdown,embed_model_dropdown):
                update_env_variable("LLM_MODEL",large_model_dropdown)
                large_model_max_tokens, embed_model_max_tokens = get_max_tokens(large_model_dropdown,embed_model_dropdown)
                return gr.update(elem_id="llm_max_tokens",value=large_model_max_tokens)
            large_model_dropdown.change(
                fn=large_model_dropdown_input,
                inputs=[large_model_dropdown,embed_model_dropdown],
                outputs=[llm_MAX_tokens],
            )
            def embed_model_dropdown_input(large_model_dropdown,embed_model_dropdown):
                update_env_variable("EMBEDDING_MODEL", embed_model_dropdown)
                large_model_max_tokens,embed_model_max_tokens = get_max_tokens(large_model_dropdown,embed_model_dropdown)
                return gr.update(elem_id="embed_max_tokens",value=embed_model_max_tokens)
            embed_model_dropdown.change(
                fn=embed_model_dropdown_input,
                inputs=[large_model_dropdown,embed_model_dropdown],
                outputs=[embed_MAX_tokens],
            )
            Thesidebar.load(
                fn=get_initial_values,
                inputs=[],
                outputs=[base_url_input, api_key_input,API_port_input]
            )

            # `closed_sidebar` 定义
        with gr.Row(elem_classes="Closed-SideBar", visible=False) as closed_sidebar:
            #gr.Markdown("侧边栏",elem_id="ASideBar")
            open_button = gr.Button("🔓 打开侧边栏", elem_id="Closed-SideBar-button")

            # 状态更新函数

            def toggle_sidebar():
                # JS 脚本：切换 sidebar 和 closed_sidebar 的显示状态
                return gr.update(elem_classes="Closed-SideBar",visible=True), gr.update(elem_classes="SideBar",visible=False)

            def toggle_back_sidebar():
                # JS 脚本：切换 sidebar 和 closed_sidebar 的显示状态
                return gr.update(elem_classes="SideBar",visible=True), gr.update(elem_classes="Closed-SideBar",visible=False)

            # 按钮点击事件

        close_button.click(fn=toggle_sidebar, outputs=[closed_sidebar, SideBar])
        open_button.click(fn=toggle_back_sidebar, outputs=[SideBar, closed_sidebar])
    return Thesidebar

def welcome_page():
    """创建欢迎使用页面"""
    with gr.Blocks(visible=False, elem_id="welcome-page") as welcome_page:
        # 标题
        gr.Markdown("# 欢迎使用", elem_id="welcome-title", elem_classes="center-text")

        # 主体内容
        with gr.Row():
            # 左侧 README 内容块
            with gr.Column(scale=3):
                gr.Markdown(load_readme(), label="项目简介")

            # 右侧状态栏
            with gr.Column(scale=1):
                gr.Markdown("## 系统状态")
                dependency_status = gr.Textbox(
                    label="依赖包状态",
                    value=check_dependency_status()[0],
                    interactive=False,
                    placeholder="依赖包安装状态显示在此处"
                )
                missing_packages_dropdown = gr.Dropdown(
                    label="缺失依赖包列表",
                    choices=[],
                    visible=True,
                    interactive=False,
                    multiselect=True,
                    allow_custom_value=True,
                    elem_id="Missing_packages_dropdown"
                )
                install_button = gr.Button(
                    "安装缺失的依赖包",
                    visible=False,
                    variant="primary",
                    elem_id="Install_button"
                )
                with gr.Column():  # 创建水平布局
                    lightrag_status = gr.Textbox(
                        label="LightRAG 后端状态",
                        value="按下方的🔄按钮以进行测试",
                        interactive=False,
                        placeholder="后端状态显示在此处"
                    )
                    with gr.Row():
                        '''
                        lightrag_fireup_button = gr.Button(
                            "💻",
                            size="sm",  # 小按钮
                            elem_id="fireup-btn",  # 为按钮设置 ID，方便样式定制
                            min_width = 100,
                        )'''
                        lightrag_status_refresh_button = gr.Button(
                            "🔄",
                            size="sm",  # 小按钮
                            elem_id="status-refresh-btn",  # 为按钮设置 ID，方便样式定制
                            min_width = 100,
                        )
                """
                ####不再使用
                model_connection_status = gr.Textbox(
                    label="大模型连接状态",
                    value=check_model_connection_status(),
                    interactive=False,
                    placeholder="模型连接状态显示在此处"
                )
                """
                api_port = gr.Textbox(
                    label="LightRAG后端地址",
                    value=check_port(),
                    interactive=False,
                    elem_id="API_port",
                    placeholder="当前的后端地址为",
                    show_copy_button=True
                )

                refresh_button = gr.Button("🔄刷新状态", variant="primary")

        # 底部链接与开源协议
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 📂 项目链接")
                gr.Markdown("""
                - [GitHub 仓库](https://github.com/HerSophia/LightRAGforSillyTavern)
                - [项目使用说明书](https://your_docs_link)
                - [视频教程](https://your_video_link)
                """)

            with gr.Column(scale=1):
                license_textbox = gr.Textbox(
                    label="开源协议",
                    value=load_license(),
                    lines=10,
                    interactive=False
                )
        # 页面初始化时的检查逻辑
        def initialize_status():
            status, missing, mismatched = check_dependency_status()
            if (len(mismatched)):
                show_missing_packages_dropdown = False
            else:
                show_missing_packages_dropdown = True
            all_issues = missing + mismatched
            show_install_button = bool(missing)  # 仅缺失包时显示安装按钮
            return (
                status,
                gr.update(elem_id="Missing_packages_dropdown",visible=show_missing_packages_dropdown),
                all_issues,
                missing,  # 控制安装按钮是否显示
                gr.update(elem_id="Install_button",visible=show_install_button)
            )

        welcome_page.load(
            fn=initialize_status,
            inputs=[],
            outputs=[
                dependency_status,
                missing_packages_dropdown,
                missing_packages_dropdown,
                missing_packages_dropdown,
                install_button,
            ],
        )

        # 刷新按钮逻辑
        def refresh_status():
            status, missing, mismatched = check_dependency_status()
            all_issues = missing + mismatched
            show_install_button = bool(missing)
            new_web = check_port()
            return (
                status,
                all_issues,
                missing,  # 控制安装按钮是否显示
                gr.update(elem_id="Install_button",visible=show_install_button),
                gr.update(elem_id="API_port"),
                new_web
            )

        def lightrag_status_refresh():
            lightrag_status = asyncio.run(check_lightrag_status())
            return lightrag_status

        lightrag_status_refresh_button.click(
            fn=lightrag_status_refresh,
            inputs=[],
            outputs=[lightrag_status]
        )
        refresh_button.click(
            fn=refresh_status,
            inputs=[],
            outputs=[
                dependency_status,
                missing_packages_dropdown,
                missing_packages_dropdown,
                install_button,
                api_port,
                api_port
            ],
        )

        # 安装按钮逻辑
        def install_and_update(missing_packages):
            if not missing_packages:
                return "没有需要安装的依赖包"
            install_result = install_missing_packages(missing_packages)
            status, _, _ = check_dependency_status()  # 检查安装后的状态
            return status, install_result

        install_button.click(
            fn=install_and_update,
            inputs=[missing_packages_dropdown],
            outputs=[
                dependency_status
            ],
        )
    return welcome_page

def file_management_ui():
    """
    创建文本文件管理页面 UI
    """
    with gr.Blocks() as file_ui:
        gr.Markdown("# 📂 文本文件管理")

        # 左侧文件列表
        with gr.Row():
            with gr.Column():
                # 显示文件列表
                file_list_output = gr.Textbox(label="文件列表", lines=15, interactive=False)
                refresh_files_button = gr.Button("🔄 刷新文件列表", variant="primary")

            # 右侧文件操作
            with gr.Column():
                selected_file = gr.Dropdown(label="选择文件", choices=[], interactive=True,multiselect=True)

                selected_file_path = gr.Textbox(label="选中的文件路径",visible=True)  # 用于记录完整路径
                with gr.Row():
                    open_text_folder_button = gr.Button("📁 打开文件夹", variant="secondary")
                    open_text_file_button = gr.Button("📄 打开文件", variant="secondary")
                with gr.Row():
                    set_env_button = gr.Button("🛠️ 设置为环境变量", variant="primary")
                    delete_text_file_button = gr.Button("🗑️ 删除文件", variant="stop")
                delete_confirmation_row = gr.Row(elem_id="Delete_confirmation_row",visible=False)
                with delete_confirmation_row:
                    confirm_delete_button = gr.Button("确认删除", variant="stop",elem_id="Confirm_delete_button")
                    cancel_delete_button = gr.Button("取消", variant="secondary")
                with gr.Row():
                    selected_file_build_graph_button = gr.Button("构建图谱")
                    selected_file_insert_graph_button = gr.Button("插入至现有图谱")
                graph_build_confirmation_row = gr.Row(elem_id="Graph_build_confirmation_row",visible=False)
                with graph_build_confirmation_row:
                    selected_confirm_build_button = gr.Button("确认构建",elem_id="Selected_Confirm_build_button")
                    selected_cancel_build_button = gr.Button("取消",elem_id="Selected_Cancel_build_button")
                graph_insert_confirmation_row = gr.Row(elem_id="Graph_insert_confirmation_row", visible=False)
                with graph_insert_confirmation_row:
                    selected_confirm_insert_button = gr.Button("确认构建", elem_id="Selected_Confirm_insert_button")
                    selected_cancel_insert_button = gr.Button("取消", elem_id="Selected_Cancel_insert_button")

                dict_selected_files = gr.Textbox(visible=False)
                operate_result = gr.Textbox(label="操作结果", interactive=False, lines=2)

        # 上传文件和操作按钮
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 上传文件")
                file_uploader = gr.File(label="上传文件", file_types=['text','.pdf','.doc','.ppt','.csv'],file_count="multiple")
                upload_button = gr.Button("上传")
                upload_result = gr.Textbox(label="上传结果", interactive=False, lines=5)

                with gr.Row():
                    with gr.Column():
                        # 为文件构建图谱、

                        upload_file_build_graph_button = gr.Button("将上传的文件构建图谱")

                        up_prebuild_dict_result = gr.Textbox(visible=False)
                        upload_file_build_confirmation_row = gr.Row(elem_id="Upload_file_build_confirmation_row",visible=False)
                        with upload_file_build_confirmation_row:
                            upload_file_confirm_build_button = gr.Button("确认构建", variant="stop")
                            upload_file_cancel_build_button = gr.Button("取消", variant="secondary")
                        build_result = gr.Textbox(label="构建结果", interactive=False, lines=5)

                    # 插入至现有图谱
                    with gr.Column():
                        upload_file_insert_graph_button = gr.Button("将上传的文件插入至现有图谱")
                        up_preinsert_dict_result = gr.Textbox(visible=False)
                        upload_file_insert_confirmation_row = gr.Row(elem_id="Upload_file_insert_confirmation_row",visible=False)
                        with upload_file_insert_confirmation_row:
                            upload_file_confirm_insert_button = gr.Button("确认插入", variant="stop")
                            upload_file_cancel_insert_button = gr.Button("取消", variant="secondary")

                        insert_result = gr.Textbox(label="插入结果", interactive=False, lines=5)


        # Tips 区域
        gr.Markdown("### ℹ️ Tips")
        gr.Markdown("""
        - **刷新文件列表**: 更新左侧文件列表。
        - **选择文件**: 在列表中选择文件进行操作。
        - **打开文件夹**: 在资源管理器中打开文件所在的文件夹。
        - **打开文件**: 使用系统默认程序打开文件。
        - **设置为环境变量**: 将文件路径设置为环境变量./files。
        - **删除文件**: 删除文件并备份。删除前会提示确认。
        - **上传文件**: 上传支持的文件类型至系统。           
        - **为该文件构建图谱**: 设置环境变量并构建知识图谱。
        - **插入至现有图谱**: 将文件内容插入当前选择的知识图谱。
        """)


        # 交互逻辑
        file_mapping = gr.State()  # 用于存储文件名和路径的映射字典
        selected_file_name = gr.Textbox(visible=False)  # 隐藏的 Textbox，用于记录选择的文件名
        selected_file_path_invisible = gr.Textbox(visible=False)
        isselected_file = []
        #debug_output = gr.Textbox(label="调试信息", lines=2, interactive=False)

        def get_file_list(file_mapping):
            file_list = refresh_dropdown_choices(file_mapping)
            return gr.update(choices=file_list,value=file_list[0] if file_list else None)

        file_ui.load(
            fn=refresh_file_list_display,
            inputs=[],
            outputs=[file_list_output, file_mapping],  # 更新文件列表并显示调试信息
        )

        file_list_output.change(
            fn=get_file_list,
            inputs=[file_mapping],
            outputs=[selected_file]
        )

        # 刷新文件列表时更新文件名列表和路径映射
        refresh_files_button.click(
            fn=refresh_file_list_display,
            inputs=[],
            outputs=[file_list_output, file_mapping],  # 更新文件列表并显示调试信息
        )

        # 更新 Dropdown 的选项
        refresh_files_button.click(
            fn=get_file_list,
            inputs=[file_mapping],
            outputs=[selected_file] # 更新 Dropdown 的选项并显示调试信息
        )

        selected_file.change(
            fn=debug_and_return,
            inputs=[selected_file],
            outputs=[selected_file_name],
        )
        def get_selected_file_path(names, mapping):
            names = eval(names)
            path = []
            dict_selected_files = {}
            for name in names:
                path.append(mapping.get(name))
                dict_selected_files[name] = mapping.get(name)
            path_textbox = "\n".join(path)
            #print(dict_selected_files)
            return path_textbox,path,dict_selected_files

        # 根据文件名查找文件路径
        selected_file_name.change(
            fn=get_selected_file_path,
            inputs=[selected_file_name, file_mapping],
            outputs=[selected_file_path,selected_file_path_invisible,dict_selected_files],  # 更新文件路径
        )
        # 按钮功能绑定
        open_text_folder_button.click(
            fn=open_text_folder,
            inputs=[selected_file_path_invisible],
            outputs=[operate_result],
        )
        open_text_file_button.click(
            fn=open_text_file,
            inputs=[selected_file_path_invisible],
            outputs=[operate_result],
        )
        set_env_button.click(
            fn=set_rag_env_variable,
            inputs=[selected_file_path_invisible],
            outputs=[operate_result],
        )
        # 按下“删除文件”按钮，显示确认删除和取消按钮
        delete_text_file_button.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[delete_confirmation_row],
        )

        # 确认删除
        confirm_delete_button.click(
            fn=delete_file_with_backup,
            inputs=[selected_file_path_invisible],
            outputs=[operate_result,delete_confirmation_row],
        )

        # 取消删除
        cancel_delete_button.click(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[delete_confirmation_row],
        )

        def selected_files_to_build():
            return f"你确定要为这些文件构造图谱吗？对应文件夹将会是./graph/（第一个文件的名字）",gr.update(elem_id="Graph_build_confirmation_row",visible=True)

        selected_file_build_graph_button.click(
            fn=selected_files_to_build,
            inputs=[],
            outputs=[operate_result,graph_build_confirmation_row],
        )
        selected_confirm_build_button.click(
            fn=build_graph_for_files,
            inputs=[dict_selected_files],
            outputs=[operate_result],
        )
        selected_cancel_build_button.click(
            fn=lambda : (f"已取消",gr.update(elem_id="Graph_build_confirmation_row",visible=False)),
            inputs=[],
            outputs=[operate_result,graph_build_confirmation_row]
        )
        # 按下“插入到现有图谱”按钮，显示确认插入和取消按钮，并提示路径信息
        selected_file_insert_graph_button.click(
            fn=lambda path: (
                f"当前选择知识图谱为 {os.getenv('RAG_DIR', '未设置')}, 你确定要插入文件 {os.path.basename(path)}?",
                gr.update(visible=True)),
            inputs=[selected_file_path_invisible],
            outputs=[operate_result, graph_insert_confirmation_row],  # 同时更新提示信息和按钮的可见性
        )
        # 确认插入
        selected_confirm_insert_button.click(
            fn=insert_graph_for_files,
            inputs=[dict_selected_files],
            outputs=[operate_result],
        )
        # 取消插入
        selected_cancel_insert_button.click(
            fn=lambda: (gr.update(value="取消插入操作。"), gr.update(visible=False)),
            inputs=[],
            outputs=[operate_result, graph_insert_confirmation_row],  # 隐藏按钮并更新提示信息
        )

        upload_button.click(
            fn=upload_files_and_save,
            inputs=[file_uploader],
            outputs=[upload_result,up_prebuild_dict_result,up_preinsert_dict_result],
        )

        upload_file_build_graph_button.click(
            fn=selected_files_to_build,
            inputs=[],
            outputs=[build_result,upload_file_build_confirmation_row]
        )
        upload_file_confirm_build_button.click(
            fn=build_graph_for_files,
            inputs=[up_prebuild_dict_result],
            outputs=[build_result]
        )
        upload_file_cancel_build_button.click(
            fn=lambda: (f"已取消", gr.update(elem_id="Upload_file_build_confirmation_row", visible=False)),
            inputs=[],
            outputs=[build_result, upload_file_build_confirmation_row]
        )

        upload_file_insert_graph_button.click(
            fn=lambda path: (
                f"当前选择知识图谱为 {os.getenv('RAG_DIR', '未设置')}, 你确定要插入这些文件?",
                gr.update(visible=True)),
            inputs=[],
            outputs=[insert_result,upload_file_insert_confirmation_row]
        )
        upload_file_confirm_insert_button.click(
            fn=build_graph_for_files,
            inputs=[up_preinsert_dict_result],
            outputs=[insert_result]
        )
        upload_file_cancel_insert_button.click(
            fn=lambda: (f"已构造", gr.update(elem_id="Upload_file_insert_confirmation_row", visible=False)),
            inputs=[],
            outputs=[insert_result, upload_file_insert_confirmation_row]
        )


    return file_ui

def graph_ui():
    """创建图谱管理页面"""
    with gr.Blocks(visible=False, elem_id="graph-page") as graph_page:  # 使用 Blocks 替代 Column
        gr.Markdown("# 📚 图谱管理页面")

        # 上部布局
        with gr.Row():
            # 左上角：文件夹列表
            with gr.Column():
                folder_list = gr.Textbox(
                    label="文件夹列表",
                    lines=22,
                    interactive=False,
                    placeholder="加载中...",
                    elem_id="folder_list"
                )
                update_folder_list_button = gr.Button(
                    "🔄刷新",min_width = 100
                )

            # 右上角：文件夹操作按钮
            with gr.Column():
                rag_folder_selector = gr.Dropdown(choices=[], label="选择文件夹")
                selected_graph_abs_path = gr.Textbox(label="选中的文件路径", elem_id="Rag_folder_selector",visible=True)  # 用于记录完整路径
                selected_graph_rel_path = gr.Textbox(label="选中的文件路径", elem_id="Rag_folder_selector",visible=False)
                with gr.Row():
                    open_button = gr.Button("📂 打开文件夹",min_width = 100)
                    open_html_button = gr.Button("📄 打开 HTML",min_width = 100)
                with gr.Row():
                    set_env_button = gr.Button("🛠️ 设为环境变量", variant="primary",min_width = 100)
                    delete_button = gr.Button("️🗑️ 删除文件夹", variant="stop",min_width = 100)
                delete_status = gr.Textbox(label="删除状态", interactive=False)
                env_status = gr.Textbox(label="环境变量状态", interactive=False)
                HTML_path = gr.Textbox(label="HTML 文件路径", interactive=False)

        # 底部：上传文件
        with gr.Row():
            with gr.Column():
                upload_zip = gr.File(label="上传 ZIP 文件", file_types=[".zip"],file_count="multiple")
                upload_button = gr.Button("上传", min_width=100)
            upload_status = gr.Textbox(label="上传状态", interactive=False,lines=9)

        # Tips 栏
        with gr.Row():
            gr.Markdown("""
            ### ℹ️ Tips:
            - **打开文件夹**: 在文件资源管理器中打开所选文件夹。
            - **打开 HTML**: 搜索所选文件夹中名为 `knowledge_graph.html` 的文件并打开。打开后会展现相应的知识图谱。
            - **设为环境变量**: 将所选文件夹路径设置为环境变量。
            - **删除文件夹**: 删除所选文件夹及其所有内容，操作不可恢复，请谨慎。
            - **上传 ZIP 文件**: 将 ZIP 文件解压至 `./graph` 的子文件夹，文件夹名与 ZIP 文件名一致。ZIP中的内容就是你的或者他人分享的图谱。
            """,
            elem_id="tips-bar",
            )

        folder_path_map = gr.State()

        # 绑定事件

        def page_load():
            folder_list, folder_path_dic, selective_list = list_subdirectories()
            return folder_list,folder_path_dic,gr.update(elem_id="Rag_folder_selector",choices=selective_list,value=selective_list[0] if selective_list else None)

        graph_page.load(
            fn=page_load,
            inputs=None,
            outputs=[folder_list,folder_path_map,rag_folder_selector]
        )

        def update_folder_list():
            folder_path_dic = {}
            folder_list,folder_path_dic,selective_list = list_subdirectories()
            #print(selective_list)
            return gr.update(elem_id="folder_list",value=folder_list),gr.update(elem_id="Rag_folder_selector",choices=selective_list,value=selective_list[0] if selective_list else None),folder_path_dic

        update_folder_list_button.click(
            fn=update_folder_list,
            inputs=[],
            outputs=[folder_list,rag_folder_selector,folder_path_map]
        )

        def mapping_path(folder_name, folder_dict):
            """
                根据文件夹名称返回对应的绝对路径和相对路径。
                :param folder_name: 要查找的文件夹名称
                :param folder_dict: 包含文件夹名称与绝对路径的字典
            """
            base_path = "./graph"
            if folder_name not in folder_dict:
                return {"error": "Folder name not found in the dictionary."}
            absolute_path = folder_dict[folder_name]
            relative_path = f"./graph/" + os.path.relpath(absolute_path, start=base_path)
            return absolute_path,relative_path

        rag_folder_selector.change(
            fn=mapping_path,
            inputs=[rag_folder_selector,folder_path_map],
            outputs=[selected_graph_abs_path,selected_graph_rel_path]
        )

        open_button.click(
            fn=open_rag_folder,
            inputs=selected_graph_abs_path,
            outputs=None,
        )

        open_html_button.click(
            fn=find_html_file,
            inputs=selected_graph_abs_path,
            outputs=HTML_path,
        )

        set_env_button.click(
            fn=set_env_variable_from_folder,
            inputs=selected_graph_rel_path,
            outputs=env_status,
        )

        delete_button.click(
            fn=backup_and_delete_graph_folder,
            inputs=[selected_graph_abs_path],
            outputs=[delete_status],
        )
        # 按钮点击逻辑：先暂存上传文件，再触发解压
        uploaded_files = gr.State([])  # 用于暂存上传的文件

        upload_zip.upload(
            fn=lambda files: files,  # 暂存上传的文件
            inputs=upload_zip,
            outputs=[uploaded_files]
        )

        upload_button.click(
            fn=process_uploaded_zips_with_progress,  # 统一处理上传的 ZIP 文件
            inputs=[uploaded_files],
            outputs=[upload_status]
        )


        upload_status.change(
            fn=update_folder_list,
            inputs=[],
            outputs=[folder_list,rag_folder_selector,folder_path_map]
        )



    return graph_page

def pdf_management_ui():
    """创建 PDF 管理页面"""
    with gr.Blocks() as ui:  # 主框架
        # 定义状态变量
        pdf_page_visible = gr.State(value=not should_show_notification() and STATE.get("dependencies_installed", False))
        notification_page_visible = gr.State(value=not pdf_page_visible.value)

        # PDF 管理页面
        with gr.Accordion(visible=pdf_page_visible.value, elem_id="pdf-management-page") as pui:
            gr.Markdown("# 🌐 HTML to PDF 转换工具")

            # 顶部布局
            with gr.Row():
                # 左侧：URL输入和列表显示
                with gr.Column():
                    gr.Markdown("### 🌍 网页地址")
                    url_input = gr.Textbox(
                        label="输入网页地址（每行一个）",
                        lines=5,
                        placeholder="请输入一个或多个网址，每行一个",
                        elem_id="url-input"
                    )
                    add_button = gr.Button(
                        "+ 添加到列表",
                        variant="primary",
                        elem_id="add-button",
                    )
                    urls_display = gr.Textbox(
                        label="已添加的网页",
                        lines=10,
                        interactive=False,
                        placeholder="当前未添加任何网页",
                        elem_id="url-display"
                    )
                    url_list = gr.State([])

                # 右侧：功能按钮区
                with gr.Column():
                    gr.Markdown("### 📄 PDF 操作")
                    generate_single_pdf = gr.Button(
                        "📘 生成单个 PDF",
                        variant="primary",
                    )
                    generate_multiple_pdfs = gr.Button(
                        "📚 生成多个 PDF",
                        variant="primary",
                    )
                    selected_pdf = gr.Textbox(
                        label="选择的 PDF 文件路径",
                        placeholder="请输入或选择 PDF 文件路径",
                        elem_id="pdf-path"
                    )
                    open_pdf_button = gr.Button(
                        "📂 打开 PDF",
                        variant="secondary",
                    )
                    delete_pdf_button = gr.Button(
                        "🗑️ 删除 PDF",
                        variant="stop",
                    )

            # 底部：操作结果显示
            with gr.Row():
                operation_output = gr.Textbox(
                    label="操作结果",
                    lines=5,
                    interactive=False,
                    elem_id="operation-output"
                )

            # 提示区域
            gr.Markdown("""
            ### ℹ️ Tips:
            - **添加到列表**: 将输入的网页地址加入待转换列表，支持多个 URL，一个URL一行。
            - **生成单个 PDF**: 将第一个网址转换为 PDF。
            - **生成多个 PDF**: 批量将所有网址转换为多个 PDF 文件。
            - **打开 PDF**: 使用系统默认应用打开选择的 PDF 文件。
            - **删除 PDF**: 删除选择的 PDF 文件，请谨慎操作。
            """, elem_id="tips-bar")

            def add_unique_url(input_urls, urls):
                """
                添加用户输入的多个网址到已有网址列表中，并去除重复项
                """
                # 拆分用户输入的网址列表，按换行符和逗号分割
                new_urls = [url.strip() for url in input_urls.splitlines() if url.strip()]

                # 合并新网址与已有网址
                combined_urls = urls + new_urls

                # 去重并保持顺序
                deduplicated_urls = list(dict.fromkeys(combined_urls))  # 使用 dict 保持顺序的去重方式

                # 返回更新后的列表和显示内容
                return deduplicated_urls, "\n".join(deduplicated_urls)

            # 按钮交互逻辑
            add_button.click(
                fn=add_unique_url,
                inputs=[url_input, url_list],
                outputs=[url_list, urls_display]
            )

            generate_single_pdf.click(
                fn=lambda urls: html_to_pdf([urls[0]]) if urls else "请先添加至少一个 URL",
                inputs=[url_list],
                outputs=[operation_output]
            )

            generate_multiple_pdfs.click(
                fn=lambda urls: html_to_pdf(urls) if urls else "请先添加至少一个 URL",
                inputs=[url_list],
                outputs=[operation_output]
            )

            open_pdf_button.click(
                fn=open_pdf,
                inputs=[selected_pdf],
                outputs=[operation_output]
            )

            delete_pdf_button.click(
                fn=lambda pdf: delete_pdf_with_backup([pdf]) if pdf else "请先选择一个 PDF 路径",
                inputs=[selected_pdf],
                outputs=[operation_output]
            )

        # 通知页面
        with gr.Accordion(visible=notification_page_visible.value, elem_id="notification-page") as notification_ui:
            gr.Markdown("### ⚠️ 通知：此页面功能尚未完成，目前处于不可用状态", visible=notification_page_visible.value)
            gr.Markdown("请安装相关依赖，或直接跳过此通知开始使用工具。", visible=notification_page_visible.value)
            install_btn = gr.Button(
                "安装依赖",
                variant="primary",
                visible=notification_page_visible.value,
            )
            close_btn = gr.Button(
                "跳过并开始使用",
                variant="secondary",
                visible=notification_page_visible.value,
            )
            remember_checkbox = gr.Checkbox(
                label="7 天内不再显示",
                elem_id="remember-checkbox",
                visible=notification_page_visible.value
            )

            # 安装依赖逻辑
            install_btn.click(
                fn=handle_install_dependencies,
                inputs=[],
                outputs=[notification_ui]
            )

            # 跳过逻辑：切换页面可见性
            def skip_notification(remember, pdf_visible):
                # 根据用户操作调整页面状态
                pdf_visible = True
                return pdf_visible, not pdf_visible, gr.update(visible=pdf_visible), gr.update(visible=False)

            close_btn.click(
                fn=skip_notification,
                inputs=[remember_checkbox, pdf_page_visible],
                outputs=[pdf_page_visible, notification_page_visible, pui, notification_ui]
            )

    return ui

def intro_animation():
    """
    纯 CSS 实现渐变文字和背景引导动画，并在动画结束后恢复滚动条。
    """
    html_content = """
    <style>
    body {
        margin: 0;
        overflow: hidden; /* 防止滚动条在动画期间显示 */
        animation: restoreOverflow 2s ease-in-out 4s forwards; /* 在动画结束后恢复滚动条 */
    }

    #intro-page {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: #ffffff; /* 背景为纯白 */
        z-index: 9999; /* 确保引导动画层在最前面 */
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeOut 2s ease-in-out 4s forwards; /* 延迟4秒后开始渐隐 */
    }

    #intro-text {
        font-size: 2rem;
        color: #333333;
        text-align: center; /* 居中对齐文字 */
        opacity: 0;
        animation: fadeInText 2s ease-in-out forwards; /* 文本渐显 */
    }
    
    #intro-text div {
        margin-top: 10px; /* 设置每行之间的间距 */
    }
    
    @keyframes fadeInText {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    @keyframes fadeOut {
        0% { opacity: 1; }
        100% {
            opacity: 0;
            z-index: -1; /* 最后阶段隐藏动画层 */
            display: none; /* 确保不再占用空间 */
        }
    }

    @keyframes restoreOverflow {
        0% { overflow: hidden; }
        100% { overflow: auto; } /* 恢复滚动条 */
    }
    </style>
    <div id="intro-page">
        <div id="intro-text">
            <div>欢迎使用</div>
            <div>LightRAG for OpenAI Standard Frontend</div>
        </div>
    </div>
    """
    return html_content

def settings_ui():
    with gr.Blocks() as settings:
        gr.Markdown("# 关于前端")
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 前端设置面板")
                start_page = gr.Checkbox(label="跳过启动页面",elem_id="Start_page")
                frontend_port = gr.Textbox(label="前端端口",elem_id="Frontend_port")
            with gr.Column():
                gr.Markdown("## 关于我们")
                who_we_are_textbox = gr.Textbox(
                    label="我们来自？",
                    value="我们算不上什么正儿八经的团队，更像是一群闲的没事干的爱好者。其中不少是大学生，因此本项目的更新会比较慢，请谅解。",
                    lines=3,
                    interactive=False
                )
                community_textbox = gr.Textbox(
                    label="想要交流？",
                    value=
                          "QQ群：xxx\n"
                          "Discord：xxx",
                    lines=2,
                    interactive=False
                )

        def start_page_show(bool_start_page):
            skip = True if bool_start_page else False
            update_env_variable("start_page_IsNotShow", str(skip))
            return gr.update(elem_id="Start_page",value = skip)

        def check_settings():
            IsNotShow = os.getenv("start_page_IsNotShow","") == 'True'
            the_frontend_port = os.getenv("FRONTEND_PORT","")
            return gr.update(elem_id="Start_page",value = IsNotShow),gr.update(elem_id="Frontend_port",value = the_frontend_port)

        settings.load(
            fn=check_settings,
            outputs=[start_page,frontend_port]
        )

        start_page.change(
            fn=start_page_show,
            inputs=[start_page]
        )
        def Frontend_port(port):
            update_env_variable("FRONTEND_PORT",port)
            return None

        frontend_port.change(
            fn=Frontend_port,
            inputs=[frontend_port]
        )
        return  settings

'''

def switch_page(page):
    """根据页面状态返回更新"""
    if page == "env_management":
        return (
            gr.update(visible=True),  # 环境变量页面可见
            gr.update(visible=False),  # 文件管理页面隐藏
            gr.update(visible=False),  # 图谱管理页面隐藏
        )
    elif page == "file_management":
        return (
            gr.update(visible=False),  # 环境变量页面隐藏
            gr.update(visible=True),  # 文件管理页面可见
            gr.update(visible=False),  # 图谱管理页面隐藏
        )
    elif page == "graph_management":
        return (
            gr.update(visible=False),  # 环境变量页面隐藏
            gr.update(visible=False),  # 文件管理页面隐藏
            gr.update(visible=True),  # 图谱管理页面可见
        )
    # 默认隐藏所有页面
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )
# 更新导航栏和主界面
def create_navbar():
    """创建导航栏"""
    with gr.Row():
        env_button = gr.Button("环境变量管理", variant="secondary", elem_id="env-btn")
        file_button = gr.Button("文本文件管理", variant="secondary", elem_id="file-btn")
        graph_button = gr.Button("知识图谱管理", variant="secondary", elem_id="graph-btn")
    return env_button, file_button, graph_button


def build_ui():
    """主界面构建"""
    with gr.Blocks() as ui:
        current_page = gr.State("file_management")  # 初始页面状态

        # 导航栏
        env_button, file_button, graph_button = create_navbar()

        # 页面容器
        with gr.Row():
            with gr.Column(visible=False, elem_id="env-page") as env_page:
                env_variables_ui()
            with gr.Column(visible=True, elem_id="file-page") as file_page:
                create_file_upload_ui()
            with gr.Column(visible=False, elem_id="graph-page") as graph_page:
                create_graph_ui()

        # 点击按钮更新页面状态
        env_button.click(
            fn=lambda: "env_management",
            inputs=None,
            outputs=current_page,
        )
        file_button.click(
            fn=lambda: "file_management",
            inputs=None,
            outputs=current_page,
        )
        graph_button.click(
            fn=lambda: "graph_management",
            inputs=None,
            outputs=current_page,
        )

        # 页面状态变更时更新可见性
        current_page.change(
            fn=switch_page,
            inputs=current_page,
            outputs=[env_page, file_page, graph_page],
        )

    return ui
'''

def build_ui_with_tabs():
    # 自定义CSS
    custom_css = """
            .SideBar {
                width: auto !important;
                height: 100% !important;
                max-width: 25% !important;
                background-color: #f5f5f5;
                padding: 10px;
                box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
            }

            .Closed-SideBar {
                width: 50% !important;
                height: 100% !important;
                max-width: 5% !important;
                background-color: #f5f5f5;
                box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
                text-align: right; /* 将内容靠右对齐 */
            }
            
            #Closed-SideBar-button {
                width: 30% !important;
                height: 50% !important;
                max-width: 5% !important;
                background: linear-gradient(90deg, #4caf50, #8bc34a);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 1rem;
                cursor: pointer;
                transition: background 0.3s ease-in-out;
                text-align: right;
            }
            
            .gradient-button {
                background: linear-gradient(90deg, #4caf50, #8bc34a);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 1rem;
                cursor: pointer;
                transition: background 0.3s ease-in-out;
            }

            .gradient-button:hover {
                background: linear-gradient(90deg, #8bc34a, #4caf50);
            }
            #admin-page-title {
                text-align: center; /* 居中对齐文本 */
                font-size: 24px; /* 调整字体大小 */
                font-weight: bold; /* 可选：使文本加粗 */
            }
            #ASideBar {
                width: auto !important;
                height: 20% !important;
                max-width: 40% !important;
                text-align: center; /* 居中对齐 */
                font-size: 40px; /* 字体大小 */
                font-weight: bold; /* 加粗 */
                background-color: #f5f5f5; /* 背景色与侧边栏一致 */
                padding: 15px; /* 内边距 */
                border-radius: 5px; /* 圆角 */
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); /* 阴影效果 */
                margin-bottom: 20px; /* 下边距 */
                color: #333; /* 字体颜色 */
            }
    """
    """构建带有 Tabs 的主界面"""

    with gr.Blocks(css=custom_css) as ui:
        def get_intro_animation():
            '''
            回调函数判断是否显示启动页面
            '''
            load_dotenv()
            Start_page_IsNotShow = os.getenv('start_page_IsNotShow', 'False').lower() == 'true'
            if not Start_page_IsNotShow:
                return intro_animation()
            return ""

        gr.HTML(get_intro_animation)

        with gr.Column():
            gr.Markdown("# 管理界面",elem_id="admin-page-title")
            with gr.Row():
                sidebar_ui()

            # 使用 Tabs 创建导航栏
                with gr.TabItem("欢迎使用"):
                    welcome_page()  # 欢迎页面

                with gr.TabItem("文件管理"):
                    file_management_ui()  # 文件管理页面

                with gr.TabItem("图谱管理"):
                    graph_ui()  # 图谱管理页面

                with gr.TabItem("HTML to Graph"):
                    pdf_management_ui()

                with gr.TabItem("关于前端"):
                    settings_ui()




    return ui


# 启动 Gradio 应用
if __name__ == "__main__":
    load_dotenv(override=True)
    F_port = int(os.getenv("FRONTEND_PORT",""))
    build_ui_with_tabs().launch(server_port=F_port, share=False)
    sleep(5)
    webbrowser.open(f"http://127.0.0.1:{F_port}")
    #asyncio.run(fetch_model_info(os.getenv("OPENAI_BASE_URL", ""),os.getenv("OPENAI_API_KEY", "")))
