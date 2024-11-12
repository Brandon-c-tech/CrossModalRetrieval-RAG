import os
from datetime import datetime
from collections import defaultdict

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
print("已加载嵌入模型")

vector_store = Chroma(
    collection_name="cell_biology_collection_5th",
    embedding_function=embeddings,
    persist_directory="/root/RAG-test/CrossModalRetrieval-RAG/assets/chroma_langchain_db",
)
print("已创建/加载向量存储")

# 使用函数进行搜索并记录得分分布
query = "磷酸激酶在细胞信号通路中的有什么用"

# 对textbook来源进行相似性搜索，返回得分最高的4个结果
results_textbook = vector_store.similarity_search_with_score(
    query,
    k=4,
    filter={"source": "textbook"}
)

# 对image来源进行相似性搜索，返回得分最高的2个结果
results_image = vector_store.similarity_search_with_score(
    query,
    k=2,
    filter={"source": "image"}
)

# 创建一个字典来存储合并后的metadata
merged_metadata = defaultdict(lambda: {"sections": defaultdict(set)})

# 处理来自textbook的结果
for res, score in results_textbook:
    chapter = res.metadata.get("chapter")
    section = res.metadata.get("section")
    subsection = res.metadata.get("subsection")
    
    # 合并metadata
    if chapter and section:
        merged_metadata[chapter]["sections"][section].add(subsection)

# 打印合并后的结果，处理空值情况
for chapter, sections in merged_metadata.items():
    print(chapter)
    for section, subsections in sections["sections"].items():
        if section:  # 确保section不为空
            print(f"  {section}")
            for subsection in subsections:
                if subsection:  # 确保subsection不为空
                    print(f"    {subsection}")

# 处理来自image的结果
for res, score in results_image:
    print(f"{res.metadata['image_name']}")

print("搜索完成")