<center><h2>🚀 LightRAG: For OpenAI Response Standard</h2></center>


![image.png](https://s2.loli.net/2024/12/18/uQAtk4C7Zm9IXVF.png)

This repository hosts the code of LightRAG. The structure of this code is based on [nano-graphrag](https://github.com/gusye1234/nano-graphrag).
![image.png](https://s2.loli.net/2024/12/18/yi1chsHWmCXTNrA.png)
</div>

### 写在最开头
非常感谢香港大学团队（[@HKUDS](https://github.com/HKUDS "@HKUDS")）的工作，没有他们就没有这一切。没有他们的无私开源，就不会有现在这个项目——我们只是在他们开垦的一片天地中上做了点微小的工作。

### 本项目的目标
#### 1.支持OpenAI标准响应格式，以便将LightRAG集成至更多地方
#### 2.方便管理RAG系统

### 本项目适用于？
#### 想为大模型搭建高效准确的知识库，但是不需要多余的功能，内存不敏感，低并发条件

### 我们做了什么？
#### 1.将响应格式写为OpenAI标准响应格式，并且支持Prefill
#### 2.多文件的构建与插入图谱
#### 3.热切换模型，自动填入模型名称与Max_tokens
#### 4.更灵活的上下文处理策略（有待完善，已经在代码里写好）
#### 5.简单但全面的前端，方便管理RAG系统：文档上传，图谱管理

### 前端一览
![image.png](https://s2.loli.net/2024/12/18/XDjilpqvQBruVtE.png)

![image.png](https://s2.loli.net/2024/12/18/T7U6brEgpstDR3q.png)

![image.png](https://s2.loli.net/2024/12/18/CLW7dhxicOepyFR.png)

### 未来？
#### 1.SubSetting：添加次要设置，为用户实现更具性价比的图谱构建
#### 2.More Prompt：实时切换Prompt
#### 3.HTML to Graph
#### And more...

### 如何使用？
#### 0.如果你会使用诸如Anaconda的管理工具，安装步骤可以直接不用看（但别忘了下面这句）
```python
pip install -e .
```
#### 1.在项目的根目录中找到setup.bat并双击打开
#### 2.待项目的依赖包安装完毕之后，双击打开Start.bat（首次使用前需要手动在.env文件中填入相对应的信息，否则无法启动）
#### 3.待两个cmd窗口都打开并运行之后，可在浏览器进入http://127.0.0.1:8848，便能看到管理前端
#### 4.在前端的侧边栏内填入相关信息，填入完毕即可刷新模型信息
#### 5.在文件管理这一页中对你的文件进行管理。构建图谱将会构建全新的一个图谱，插入图谱则是会将文件中的信息插入至已有的图谱中。
#### 6.在图谱管理页面，记得将生成的图谱文件夹或者上传之后新建的文件夹设置为环境变量
#### 7.一切就绪，现在可以将LightRAG的后端地址填入你所使用的任何大模型问答/聊天前端。

### 也许会有点小问题？
由于我们才粗学浅，本项目不可避免地会有各种奇奇怪怪的bug，就连功能解耦都没有写好。所以遇到了bug请尽量告诉我们。

### 关于我们
关于我们自己......没什么好说的，大多是学生。所以如果有什么建议甚至是Pull Request，我们很欢迎。

## Contribution

Thank you to all our contributors!

<a href="https://github.com/HKUDS/LightRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/LightRAG" />
</a>

## 🌟Citation

```python
@article{guo2024lightrag,
title={LightRAG: Simple and Fast Retrieval-Augmented Generation},
author={Zirui Guo and Lianghao Xia and Yanhua Yu and Tu Ao and Chao Huang},
year={2024},
eprint={2410.05779},
archivePrefix={arXiv},
primaryClass={cs.IR}
}
```
**Thank you for your interest in our work!**
