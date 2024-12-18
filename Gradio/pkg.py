import pkg_resources

def export_dependencies(output_file):
    """
    将当前虚拟环境的所有依赖包及其版本号导出到指定的 txt 文件中。
    :param output_file: 输出文件的路径
    """
    try:
        # 获取所有已安装的依赖包
        installed_packages = pkg_resources.working_set

        # 按包名排序，生成格式化字符串
        dependencies = sorted([f"{pkg.key}=={pkg.version}" for pkg in installed_packages])

        # 将结果写入到指定文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(dependencies))
        print(f"依赖包列表已成功导出至 {output_file}")
    except Exception as e:
        print(f"导出依赖包列表时出错: {e}")

# 示例用法
if __name__ == "__main__":
    output_file = "./dependencies.txt"  # 输出文件路径
    export_dependencies(output_file)