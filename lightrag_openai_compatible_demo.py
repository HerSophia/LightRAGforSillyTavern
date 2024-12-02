import asyncio
import os

import numpy as np
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc

#你的知识图谱存放的文件夹
WORKING_DIR = "./dickens"

#你的文档，例如./book.txt，建议存放在files这个文件夹以便管理
file_DIR = ""

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

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=2048,
                func=embedding_func,
            ),
        )


        with open("./斗破前十章.txt", "r", encoding="utf-8", errors='ignore') as f:
            await rag.ainsert(f.read())

        '''
        #以下是搜索方法，共四种，请查看官方文档以便选择
        # Perform naive search
        print(
            await rag.aquery(
                "What kind of story is told in this first chapter? Please answer in Chinese.", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "What kind of story is told in this first chapter? Please answer in Chinese.", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "What kind of story is told in this first chapter? Please answer in Chinese.", param=QueryParam(mode="global"),
            )
        )

        '''
        # Perform hybrid search
        print(
            await rag.aquery(
                "What kind of story is told in this first chapter? Please answer in Chinese.", param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
