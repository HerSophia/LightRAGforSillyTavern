import networkx as nx
from pyvis.network import Network
import random
import os


class KnowledgeGraphVisualizer:
    def __init__(self, graphml_path: str):
        """
        初始化 KnowledgeGraphVisualizer 类。

        :param graphml_path: 输入的 GraphML 文件路径。
        """
        self.graphml_path = graphml_path
        self.graph = None
        self.net = None
        self.html_output_path = None

    def load_graph(self):
        """
        从 GraphML 文件加载图形。
        """
        if not os.path.exists(self.graphml_path):
            raise FileNotFoundError(f"GraphML file not found at: {self.graphml_path}")

        # 加载图形
        self.graph = nx.read_graphml(self.graphml_path)

    def create_network(self):
        """
        创建 Pyvis 网络，并设置节点颜色。
        """
        # 创建一个 Pyvis 网络实例
        self.net = Network(height="100vh", notebook=True)

        # 将 NetworkX 图转为 Pyvis 网络
        self.net.from_nx(self.graph)

        # 给每个节点添加随机颜色
        for node in self.net.nodes:
            node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def save_html(self):
        """
        保存生成的网络图为 HTML 文件。
        """
        # 获取 HTML 输出路径（保存在与 GraphML 文件相同的目录）
        output_dir = os.path.dirname(self.graphml_path)  # 获取 GraphML 文件所在的目录
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

        # 使用 GraphML 文件名生成 HTML 文件名，替换扩展名为 .html
        file_name_without_ext = os.path.splitext(os.path.basename(self.graphml_path))[0]
        self.html_output_path = os.path.join(output_dir, f"knowledge_graph.html")

        # 保存并显示网络图
        self.net.show(self.html_output_path)

    def generate_graph(self):
        """
        生成知识图谱并保存为 HTML 文件。
        """
        # 加载图形
        self.load_graph()

        # 创建网络并添加颜色
        self.create_network()

        # 保存 HTML 文件
        self.save_html()

        return self.html_output_path


# 用法示例：
if __name__ == "__main__":
    # GraphML 文件路径
    graphml_path = "H:/LightRAG for Sillytavern/LightRAG-for-OpenAI-Standard-Frontend/graph/第二章.txt_20241218193114/graph_chunk_entity_relation.graphml"

    # 初始化知识图谱可视化对象
    visualizer = KnowledgeGraphVisualizer(graphml_path)

    # 生成图并保存为 HTML 文件
    html_file_path = visualizer.generate_graph()
    print(f"HTML file has been saved to: {html_file_path}")
