import json

class Entry:
    """
    定义单个 entry 的数据结构
    """
    """
    uid：uid
    key：关键词
    comment：标题
    content：条目内容
    order：顺序，以对llm影响的效果分级如下：0-角色定义之前（最弱），1-角色定义之后（次弱），2-作者注释之前（较强），3-作者注释之后（极强），4-@ D（可变），5-示例消息之前，6-示例消息之前，
    position：位置，默认为100，影响效果总体受深度限制
    depth：深度，指插入第X条消息的上方。默认为4,0为最强
    probability：概率，0为不触发
    """
    def __init__(self, uid, key, comment, content, order, position, depth, probability):
        self.uid = uid
        self.key = key
        self.comment = comment
        self.content = content
        self.order = order
        self.position = position
        self.depth = depth
        self.probability = probability

    def __str__(self):
        """
        定义打印格式
        """
        return (
            f"UID: {self.uid}\n"
            f"Key: {self.key}\n"
            f"Comment: {self.comment}\n"
            f"Content:\n{self.content}\n"
            f"Order: {self.order}\n"
            f"Position: {self.position}\n"
            f"Depth: {self.depth}\n"
            f"Probability: {self.probability}\n"
        )

class JSONManager:
    """
    管理整个 JSON 数据，提供提取和操作功能
    """
    def __init__(self, json_file_path):
        self.file_path = json_file_path
        self.entries = self._load_entries()

    def _load_entries(self):
        """
        加载 JSON 文件并转换为 Entry 实例列表
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            entries = data.get("entries", {})
            return [
                Entry(
                    uid=entry_data["uid"],
                    key=entry_data["key"],
                    comment=entry_data["comment"],
                    content=entry_data["content"],
                    order=entry_data["order"],
                    position=entry_data["position"],
                    depth=entry_data["depth"],
                    probability=entry_data["probability"]
                )
                for entry_data in entries.values()
            ]
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []

    def get_entry_by_uid(self, uid):
        """
        根据 UID 获取单个 Entry
        """
        for entry in self.entries:
            if entry.uid == uid:
                return entry
        return None

    def get_entries_by_key(self, key):
        """
        根据 key 搜索并返回匹配的 Entry 列表
        """
        return [entry for entry in self.entries if key in entry.key]

    def get_all_entries(self):
        """
        返回所有 Entry 的列表
        """
        return self.entries

class EntrySorter:
    """
    用于排序和输出条目
    """

    def __init__(self, entries):
        """
        初始化，加载 Entry 列表
        :param entries: List of Entry objects
        """
        self.entries = entries

    def filter_and_sort(self):
        """
        过滤和排序条目
        :return: 排序后的 Entry 列表
        """
        # 过滤 probability 为 0 的条目
        valid_entries = [entry for entry in self.entries if entry.probability > 0]

        # 按优先级排序：order (降序) -> depth (升序) -> position (升序)
        sorted_entries = sorted(
            valid_entries,
            key=lambda x: (-x.order, x.depth, x.position)
        )
        return sorted_entries

    def export_to_txt(self, output_file):
        """
        导出排序结果到 txt 文件
        :param output_file: 输出文件路径
        """
        sorted_entries = self.filter_and_sort()

        with open(output_file, 'w', encoding='utf-8') as file:
            for entry in sorted_entries:
                file.write(f"{entry.comment} - {','.join(entry.key)} - {entry.content}\n")
        print(f"排序后的条目已保存到 {output_file}")


# 示例调用
if __name__ == "__main__":
    # 替换为您的 JSON 文件路径
    file_path = "./斗罗.json"

    # 初始化 JSONManager
    manager = JSONManager(file_path)

    # 获取所有条目
    all_entries = manager.get_all_entries()


    # 创建 EntrySorter 并导出排序结果
    sorter = EntrySorter(all_entries)
    output_path = "sorted_entries.txt"
    sorter.export_to_txt(output_path)

    """
    print("所有条目:")
    for entry in all_entries:
        print(entry)
        print("-" * 30)

    # 根据 UID 获取条目
    uid = 1
    specific_entry = manager.get_entry_by_uid(uid)
    print(f"UID 为 {uid} 的条目:")
    print(specific_entry)

    # 根据 Key 搜索条目
    search_key = "44"
    matching_entries = manager.get_entries_by_key(search_key)
    print(f"Key 包含 '{search_key}' 的条目:")
    for entry in matching_entries:
        print(entry)
    """
