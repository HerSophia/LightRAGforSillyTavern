import json
import os


def format_json_file(input_file_path, output_file_path):
    """
    读取 JSON 文件，格式化其内容并写入到一个新文件。

    :param input_file_path: 输入 JSON 文件的相对路径
    :param output_file_path: 输出格式化 JSON 文件的相对路径
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file_path):
            print(f"文件 {input_file_path} 不存在！")
            return

        # 读取 JSON 文件内容
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            data = json.load(input_file)

        # 格式化 JSON 内容
        formatted_json = json.dumps(data, indent=4, ensure_ascii=False)

        # 输出格式化内容到新文件
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(formatted_json)

        print("格式化后的 JSON 内容：")
        print(formatted_json)
        print(f"\n格式化后的内容已保存到 {output_file_path}")

    except json.JSONDecodeError as e:
        print(f"JSON 解码错误：{e}")
    except Exception as e:
        print(f"发生错误：{e}")


# 示例调用
input_file = "./test1.json"  # 输入文件路径
output_file = "./formatted_output.json"  # 输出文件路径
format_json_file(input_file, output_file)
