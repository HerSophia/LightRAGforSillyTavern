import mimetypes
import os
import shutil
import sys
import zipfile
import subprocess

from importlib.metadata import distributions
from typing import List, Tuple
from datetime import datetime, time, timedelta

import gradio as gr
from dotenv import load_dotenv, set_key
from pathlib import Path

from fastapi import requests

from playwright.sync_api import sync_playwright
from sympy import false

load_dotenv()

# 欢迎界面<开始>
class welcome_pages:

    def __init__(self):
        """初始化并构建欢迎页面"""
        self.ui = self.build_welcome_page()

    def load_readme(self):
        """加载 README.md 内容"""
        try:
            with open("README.md", "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return "README.md 文件未找到，请检查项目目录。"

    def load_license(self):
        """加载开源协议内容"""
        try:
            with open("LICENSE", "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return "LICENSE 文件未找到，请检查项目目录。"

    def load_requirements(self):
        """读取 requirements.txt 中的依赖包信息"""
        with open("requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        return [pkg.split("==")[0] for pkg in requirements], requirements

    def check_installed_packages(self):
        """获取当前环境中已安装的依赖包及版本"""
        installed_packages = {dist.metadata["Name"].lower(): dist.version for dist in distributions()}
        return installed_packages

    def check_dependency_status(self):
        """检查依赖包状态"""
        required_packages, full_requirements = self.load_requirements()
        installed_packages = self.check_installed_packages()

        missing_packages = []
        mismatched_versions = []

        for req in full_requirements:
            pkg, _, version = req.partition("==")
            pkg_lower = pkg.lower()
            if pkg_lower not in installed_packages:
                missing_packages.append(f"🚫 {req}")
            elif installed_packages[pkg_lower] != version:
                mismatched_versions.append(
                    f"⚠️ {pkg} (expected {version}, found {installed_packages[pkg_lower]})"
                )

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
    def install_and_update(missing_packages, self=None):
        if not missing_packages:
            return "没有需要安装的依赖包"
        install_result = self.install_missing_packages(missing_packages)
        status, _, _ = self.check_dependency_status()  # 检查安装后的状态
        return status, install_result


    def check_lightrag_status(self):
        """检查 LightRAG 后端状态"""
        # 示例实现，可以扩展为实际后端服务的检查逻辑
        return "✅LightRAG 后端运行正常"

    def check_model_connection_status(self):
        """检查大模型连接状态"""
        # 示例实现，可以扩展为实际模型连接的检查逻辑
        return "✅大模型连接成功"


    # 刷新按钮逻辑
    def refresh_status(self):
        status, missing, mismatched = self.check_dependency_status()
        return (
            status,
            missing + mismatched,  # 展示所有缺失和版本问题
            bool(missing or mismatched),
        )

    # 欢迎界面<结束>


    # UI

    def build_welcome_page(self):
        """创建欢迎使用页面"""
        with gr.Blocks(visible=False, elem_id="welcome-page") as welcome_page:
            # 标题
            gr.Markdown("# 欢迎使用", elem_id="welcome-title", elem_classes="center-text")

            # 主体内容
            with gr.Row():
                # 左侧 README 内容块
                with gr.Column(scale=3):
                    gr.Markdown(self.load_readme(), label="项目简介")

                # 右侧状态栏
                with gr.Column(scale=1):
                    gr.Markdown("## 系统状态")
                    dependency_status = gr.Textbox(
                        label="依赖包状态",
                        value=self.check_dependency_status()[0],
                        interactive=False,
                        placeholder="依赖包安装状态显示在此处"
                    )
                    missing_packages_dropdown = gr.Dropdown(
                        label="缺失依赖包列表",
                        choices=[],
                        visible=True,
                        interactive=False,
                        multiselect=True,
                        allow_custom_value=True
                    )
                    install_button = gr.Button(
                        "安装缺失的依赖包",
                        visible=False,
                        variant="primary",
                    )
                    lightrag_status = gr.Textbox(
                        label="LightRAG 后端状态",
                        value=self.check_lightrag_status(),
                        interactive=False,
                        placeholder="后端状态显示在此处"
                    )
                    model_connection_status = gr.Textbox(
                        label="大模型连接状态",
                        value=self.check_model_connection_status(),
                        interactive=False,
                        placeholder="模型连接状态显示在此处"
                    )
                    refresh_button = gr.Button("🔄刷新状态", variant="primary")

            # 底部链接与开源协议
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 📂 项目链接")
                    gr.Markdown("""
                    - [GitHub 仓库](https://github.com/your_repo)
                    - [项目使用说明书](https://your_docs_link)
                    - [视频教程](https://your_video_link)
                    """)

                with gr.Column(scale=1):
                    license_textbox = gr.Textbox(
                        label="开源协议",
                        value=self.load_license(),
                        lines=10,
                        interactive=False
                    )
            # 页面初始化时的检查逻辑
            def initialize_status():
                status, missing, mismatched = self.check_dependency_status()
                all_issues = missing + mismatched
                show_install_button = bool(missing)  # 仅缺失包时显示安装按钮
                return (
                    status,
                    all_issues,
                    missing,  # 控制安装按钮是否显示
                    show_install_button,
                )

            welcome_page.load(
                fn=initialize_status,
                inputs=[],
                outputs=[
                    dependency_status,
                    missing_packages_dropdown,
                    missing_packages_dropdown,
                    install_button,
                ],
            )

            # 刷新按钮逻辑
            def refresh_status():
                status, missing, mismatched = self.check_dependency_status()
                all_issues = missing + mismatched
                return (
                    status,
                    all_issues,
                    missing,  # 控制安装按钮是否显示
                    bool(missing),
                )

            refresh_button.click(
                fn=refresh_status,
                inputs=[],
                outputs=[
                    dependency_status,
                    missing_packages_dropdown,
                    missing_packages_dropdown,
                    install_button,
                ],
            )

            # 安装按钮逻辑
            def install_and_update(missing_packages):
                if not missing_packages:
                    return "没有需要安装的依赖包"
                install_result = self.install_missing_packages(missing_packages)
                status, _, _ = self.check_dependency_status()  # 检查安装后的状态
                return status, install_result

            install_button.click(
                fn=install_and_update,
                inputs=[missing_packages_dropdown],
                outputs=[
                    dependency_status,
                    gr.Textbox(placeholder="安装状态", interactive=False),
                ],
            )
        return welcome_page