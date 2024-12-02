
import os
import secrets
import time

import httpx
import numpy as np
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding, openai_compatible_complete_if_cache, \
    openai_compatible_embedding
from lightrag.utils import EmbeddingFunc
from fastapi import FastAPI, HTTPException, File, UploadFile, requests
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union


from fastapi.responses import StreamingResponse

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"
app = FastAPI(title="LightRAG API", description="API for RAG operations")

load_dotenv()

# Configure working directory
WORKING_DIR = os.getenv("RAG_DIR")
print(f"WORKING_DIR: {WORKING_DIR}")
LLM_MODEL = os.getenv("LLM_MODEL")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE"))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")

BASE_URL=os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

available_models = []

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# LLM model function
"""
在本项目中，llm_model_func与embedding function默认使用openai_compatible的相关代码，如有需要可在/lightrag/llm.py寻找到你想要使用对应的代码，以配合你使用的llm
如果真的需要修改的话，请在/lightrag/lightrag.py中的llm_model_func和embedding function进行相对应的修改
"""

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_compatible_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        **kwargs,
    )


# Embedding function


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_compatible_embedding(
        texts,
        EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    #print(f"{embedding_dim=}")
    return embedding_dim


# Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=asyncio.run(get_embedding_dim()),
        #embedding_dim=768,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=embedding_func,
    ),
)


# Data models(LightRAG standard)


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False


class InsertRequest(BaseModel):
    text: str


class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None


# Data models(OpenAI standard)

# 消息模型
class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

# 请求模型
class ChatRequest(BaseModel):
    model: str  # 模型名称
    messages: List[Message]  # 消息历史
    temperature: Optional[float] = 1.0  # 可选，生成的随机性
    top_p: Optional[float] = 1.0  # 可选，nucleus 采样
    n: Optional[int] = 1  # 可选，返回生成结果的数量
    stream: Optional[bool] = False  # 是否以流式传输返回
    stop: Optional[Union[str, List[str]]] = None  # 停止生成的标记
    max_tokens: Optional[int] = None  # 生成的最大 token 数量
    presence_penalty: Optional[float] = 0.0  # 可选，基于 token 出现的惩罚系数
    frequency_penalty: Optional[float] = 0.0  # 可选，基于 token 频率的惩罚系数
    user: Optional[str] = None  # 可选，用户标识

# 选项模型
class Choice(BaseModel):
    index: int  # 结果索引
    message: Message  # 每个结果的消息
    finish_reason: Optional[str]  # 生成结束的原因，例如 "stop"

# 使用统计模型
class Usage(BaseModel):
    prompt_tokens: int  # 提示词 token 数
    completion_tokens: int  # 生成的 token 数
    total_tokens: int  # 总 token 数

# 响应模型
class ChatCompletionResponse(BaseModel):
    id: str  # 响应唯一 ID
    object: str  # 响应类型，例如 "chat.completion"
    created: int  # 响应创建的时间戳
    model: str  # 使用的模型名称
    choices: List[Choice]  # 生成的结果列表
    usage: Optional[Usage]  # 可选，使用统计信息

# 历史消息的处理策略
def process_messages(
    user_message: str,
    system_prompt: Optional[str],
    history_messages: list[dict],
    strategy: str = "full_context",
) -> str:
    """
    处理消息的方法，用于生成最终需要传递给 RAG 的输入。

    Args:
        user_message (str): 当前用户消息。
        system_prompt (Optional[str]): 系统提示。
        history_messages (list[dict]): 多轮对话历史记录。
        strategy (str): 消息处理策略，默认为 "full_context"。

    Returns:
        str: 处理后的完整输入消息。
    """
    if strategy == "current_only":
        # 仅处理当前用户输入，不添加上下文
        return user_message

    elif strategy == "recent_context":
        # 仅保留最近几轮上下文
        recent_messages = history_messages[-3:]  # 最近 3 条对话
        full_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in recent_messages
        )
        return f"{full_context}\nUser: {user_message}"

    elif strategy == "full_context":
        # 完整上下文处理
        full_context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history_messages
        )
        return f"System: {system_prompt}\n{full_context}\nUser: {user_message}" if system_prompt else f"{full_context}\nUser: {user_message}"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def append_random_hex_to_list(user_messages: list, hex_length: int = 8) -> list:
    """
    在用户消息列表的每一项末尾添加一个随机的十六进制数。

    Args:
        user_messages (list): 原始用户消息的列表。
        hex_length (int): 生成的十六进制数长度，默认为8。

    Returns:
        list: 添加了随机十六进制数后的用户消息列表。
    """
    modified_messages = []
    for message in user_messages:
        if isinstance(message, str):  # 确保每个项是字符串
            random_hex = secrets.token_hex(nbytes=hex_length // 2)
            modified_messages.append(f"{message}\n\n\n---The following strings are for markup only and are not relevant to the above, so please ignore them---\n\n\n[#{random_hex}]")
        else:
            modified_messages.append(message)  # 非字符串项原样返回
    return modified_messages


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions_endpoint(request: ChatRequest):
    """
    QueryParam:
        mode:处理模式，共四种，请查看官方文档以便选择
        only_need_context：在本项目中，即使为true也不影响上下文策略
    """
    try:
        asyncio.run(get_model_info())

        # 确认使用何种模型
        if request.model not in available_models:
            raise HTTPException(status_code=400, detail="Selected model is not available.")
        else:
            llm_model = request.model

        # 全局变量更新
        global LLM_MODEL
        LLM_MODEL  = llm_model


        # Extract user query from messages
        user_message = [msg.content for msg in request.messages if msg.role == "user"]

        # 添加字符串以绕过hash碰撞
        appended_message = append_random_hex_to_list(user_message,8)

        user_messages = "\n".join(appended_message)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found.")


        # Extract system prompt and history messages
        system_prompts = [
            msg.content for msg in request.messages if msg.role == "system"
        ]
        system_prompt = "\n".join(system_prompts)


        history_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
            if msg.role in ["user", "assistant"]
        ]

        processed_message = process_messages(
            user_message=user_messages,
            system_prompt=system_prompt,
            history_messages=history_messages,
            strategy="full_context",  # 默认使用完整上下文策略
        )

        # Simulate RAG query result（不再使用）
        async def simulate_rag_query(query, system_prompt, history):
            # Simulated result, replace with actual rag.query call
            await rag.query(
                processed_message,
                param=QueryParam(mode="hybrid", only_need_context=False),
                #addon_params={"language": "Simplified Chinese"},
                #system_prompt_from_frontend=system_prompt,  # 添加 system_prompt
                #history_messages=history_messages,  # 添加 history_messages
            )
            return f"Simulated response to '{query}'"

        # Stream generator
        async def stream_generator():
            result = rag.query(
                processed_message,
                param=QueryParam(mode="local", only_need_context=False),
                system_prompt_from_frontend=system_prompt,  # 添加 system_prompt
            )
            content_chunks = result.split()
            for chunk in content_chunks:
                yield f'data: {{"id": "chunk", "object": "chat.completion.chunk", "choices": [{{"index": 0, "delta": {{"content": "{chunk}"}}}}]}}\n\n'
                await asyncio.sleep(0.1)

            yield 'data: {"id": "done", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'

        # Stream or standard response
        if request.stream:
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            result = rag.query(
                processed_message,
                param=QueryParam(mode="local", only_need_context=False),
                system_prompt_from_frontend=system_prompt,  # 添加 system_prompt
            )
            created_time = int(time.time())
            """
            print("\n\n\n")
            print(f"prompt_tokens: {len(system_prompt.split())}")
            print(f"user_message: {len(processed_message.split())}")
            """
            return ChatCompletionResponse(
                id="completion",
                object="chat.completion",
                created=created_time,
                model=llm_model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=result),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(system_prompt.split()),
                    completion_tokens=len(result.split()),
                    total_tokens=len(system_prompt.split()) + len(result.split()),
                ),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def get_model_info():
    global available_models
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    async with httpx.AsyncClient() as client:
        try:
            # 请求第三方 API
            response = await client.get(BASE_URL + f"/models", headers=headers)
            response.raise_for_status()  # 如果状态码非 2xx 会抛出异常

            # 获取响应数据并提取模型名称
            response_data = response.json()
            available_models = [model["id"] for model in response_data.get("data", [])]

            # 将 JSON 数据直接中转返回
            return JSONResponse(
                content=response.json(),  # 使用解析后的 JSON 数据
                status_code=response.status_code  # 保留原始响应的状态码
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="API request failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# LightRAG standard response
@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode, only_need_context=request.only_need_context
                ),
            ),
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert_file", response_model=Response)
async def insert_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        # Read file content
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            content = file_content.decode("gbk")
        # Insert file content
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))

        return Response(
            status="success",
            message=f"File content from {file.filename} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


async def test_completion():
    # 构造一个简单的测试 Prompt
    prompt = "What are the key features of FastAPI?"
    user_message = "Hi!"

    # 构建 RAG 实例（确保参数正确）
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,  # 使用您之前定义的 llm_model_func
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=2048,
            func=embedding_func,
        ),
    )

    # 调用 `query` 方法
    try:
        loop = asyncio.get_event_loop()
        print("Completion result:")
        result = rag.query(
            user_message,
            param=QueryParam(
                mode="hybrid",  # 模式：可选 "hybrid", "retrieval", 或 "generation"
                only_need_context=False,  # 是否只返回上下文
            ),
        )
        print(result)
        #print(asyncio.iscoroutinefunction(rag.query))  # 输出是否为异步函数
        #return result
    except Exception as e:
        print(f"Error during completion: {e}")

# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

if __name__ == "__main__":
    import uvicorn

    #test_funcs()
    #asyncio.run(test_funcs())

    uvicorn.run(app, host="0.0.0.0", port=8020)

# Usage example
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid"}'

# 2. Insert text:
# curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "your text here"}'

# 3. Insert file:
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: application/json" -d '{"file_path": "path/to/your/file.txt"}'

# 4. Health check:
# curl -X GET "http://127.0.0.1:8020/health"
