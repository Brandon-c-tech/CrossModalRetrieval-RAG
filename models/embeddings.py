'''
该模块提供了文档加载和嵌入功能，支持从Markdown和JSON文件中提取内容并将其存储到向量数据库中。
支持的输入格式参考assets中的_figures_description.json和细胞生物学（5）.md

函数：
- process_files(collection_name: str, file_paths: List[str], parse_mode: str = "subsection", db_path: str = "./chroma_langchain_db") -> None:
    处理多个文件并将其嵌入到指定的向量数据库中。

参数：
- collection_name (str): 向量数据库的名称。
- file_paths (List[str]): 包含一个或多个文件路径的列表，支持Markdown (.md) 和 JSON (.json) 文件。
- parse_mode (str): Markdown解析模式，默认为 "subsection"，可选值包括 "none", "chapter", "section", "subsection"。
- db_path (str): 向量数据库的创建路径，默认为 "./chroma_langchain_db"。

测试模式：
- 电子课本Markdown
python /root/RAG-test/CrossModalRetrieval-RAG/models/embeddings.py \
    --test_mode \
    --file_type markdown \
    --markdown_path /root/RAG-test/CrossModalRetrieval-RAG/assets/细胞生物学（5）.md \
    --output_dir /root/RAG-test/CrossModalRetrieval-RAG/assets \
    --parse_mode subsection

- 图片描述JSON
python /root/RAG-test/CrossModalRetrieval-RAG/models/embeddings.py \
    --test_mode \
    --file_type json \
    --json_path "/root/RAG-test/CrossModalRetrieval-RAG/assets/6 细胞生物学（5）_figures_description.json" \
    --output_dir /root/RAG-test/CrossModalRetrieval-RAG/assets
'''

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from uuid import uuid4
from datetime import datetime
import argparse
import tiktoken

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class BaseDocumentLoader(ABC):
    """文档加载器的抽象基类"""
    
    @abstractmethod
    def process_item(self, item: Dict[str, Any]) -> Document:
        """处理单个数据项并转换为Document对象"""
        pass
    
    def load_documents(self, file_path: str) -> List[Document]:
        """加载并处理文件中的所有数据"""
        pass

class TextbookMarkdownLoader(BaseDocumentLoader):
    """
    电子课本Markdown文档的加载器，不同教科书格式可能不同，还需要额外处理
    """
    
    def __init__(self, debug_mode=False, debug_output_path="debug_output.json", parse_mode="subsection"):
        """
        初始化加载器
        Args:
            debug_mode: 是否启用调试模式
            debug_output_path: 调试输出文件路径
            parse_mode: 解析模式，可选值：
                - "none": 不按标题分割
                - "chapter": 只分割到章节级别
                - "section": 分割到节级别
                - "subsection": 分割到小节级别（默认）
        """
        self.debug_mode = debug_mode
        self.debug_output_path = debug_output_path
        self.parse_mode = parse_mode
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 7200
        self.chunk_size = 5000
        self.overlap_size = 500
    
    def _get_token_count(self, text: str) -> int:
        """获取文本的token数量"""
        return len(self.encoding.encode(text))
    
    def _process_long_line(self, line: str, metadata: Dict[str, Any], documents: List[Document]):
        """处理超长的单行文本"""
        start = 0
        while start < len(line):
            chunk = line[start:start + self.chunk_size]
            documents.append(self.process_item({
                "content": chunk,
                **metadata
            }))
            start += (self.chunk_size - self.overlap_size)
    
    def process_item(self, item: Dict[str, Any]) -> Document:

        # 生成当前时间戳
        timestamp = datetime.now().isoformat()

        return Document(
            page_content=item['content'],
            metadata={
                "source": "textbook",
                "chapter": item.get("chapter"),
                "section": item.get("section"),
                "subsection": item.get("subsection"),
                "subject": item.get("subject"),
                "timestamp": timestamp
            }
        )
    
    def load_documents(self, markdown_path: str) -> List[Document]:
        documents = []
        current_chapter = ""
        current_section = ""
        current_subsection = ""
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = ""
            content_level = None
            
            def append_content(content: str, metadata: Dict[str, Any], level: str):
                if not content.strip():
                    return
                    
                # 根据解析模式调整metadata
                adjusted_metadata = metadata.copy()
                if self.parse_mode == "none":
                    adjusted_metadata['chapter'] = ""
                    adjusted_metadata['section'] = ""
                    adjusted_metadata['subsection'] = ""
                elif self.parse_mode == "chapter":
                    adjusted_metadata['section'] = ""
                    adjusted_metadata['subsection'] = ""
                elif self.parse_mode == "section":
                    adjusted_metadata['subsection'] = ""
                
                if self._get_token_count(content) > self.max_tokens:
                    # 如果当前内容超过token限制，按chunk_size分割
                    start = 0
                    while start < len(content):
                        chunk = content[start:start + self.chunk_size]
                        documents.append(self.process_item({
                            "content": chunk,
                            **adjusted_metadata
                        }))
                        start += (self.chunk_size - self.overlap_size)
                else:
                    documents.append(self.process_item({
                        "content": content,
                        **adjusted_metadata
                    }))

            for line in f:
                current_metadata = {
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "subject": os.path.basename(markdown_path).replace('.md', '')
                }

                # 根据解析模式处理标题
                if self.parse_mode != "none":
                    if line.startswith('# ') and self.parse_mode in ["chapter", "section", "subsection"]:
                        append_content(content, current_metadata, content_level or 'chapter')
                        current_chapter = line[2:].strip()
                        current_section = ""
                        current_subsection = ""
                        content = ""
                        content_level = 'chapter'
                    elif line.startswith('## ') and self.parse_mode in ["section", "subsection"]:
                        append_content(content, current_metadata, content_level or 'section')
                        current_section = line[3:].strip()
                        current_subsection = ""
                        content = ""
                        content_level = 'section'
                    elif line.startswith('### ') and self.parse_mode == "subsection":
                        append_content(content, current_metadata, content_level or 'subsection')
                        current_subsection = line[4:].strip()
                        content = ""
                        content_level = 'subsection'
                    else:
                        # 检查单行是否超过token限制
                        if self._get_token_count(line) > self.max_tokens:
                            # 先处理现有content
                            append_content(content, current_metadata, content_level or 'subsection')
                            content = ""
                            # 处理超长行
                            self._process_long_line(line, current_metadata, documents)
                        else:
                            # 检查添加新行后是否会超过限制
                            new_content = content + line
                            if self._get_token_count(new_content) > self.max_tokens:
                                # 先处理现有content
                                append_content(content, current_metadata, content_level or 'subsection')
                                content = line
                            else:
                                content += line
                else:
                    content += line
            
            # 处理最后一个部分
            append_content(content, current_metadata, content_level or 'subsection')
        
        # 添加调试输出
        if self.debug_mode:
            debug_output = []
            for doc in documents:
                debug_output.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # 将结果写入JSON文件
            with open(self.debug_output_path, 'w', encoding='utf-8') as f:
                json.dump(debug_output, f, ensure_ascii=False, indent=2)
        
        return documents

class ImageDescriptionLoader(BaseDocumentLoader):
    """
    图片描述JSON的加载器，将JSON中的值作为content，键作为metadata
    所有教科书采用了相同的json格式
    """
    
    def process_item(self, item: Dict[str, Any]) -> Document:
        # 获取图片描述作为content
        image_name = list(item.keys())[0]  # 获取第一个（也是唯一的）键
        content = item[image_name]  # 获取对应的值作为content
        
        # 生成当前时间戳
        timestamp = datetime.now().isoformat()
        
        return Document(
            page_content=content,
            metadata={
                "image_name": image_name,
                "timestamp": timestamp
            }
        )
    
    def load_documents(self, json_path: str) -> List[Document]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        # 遍历JSON对象的每一项
        for key, value in data.items():
            doc = self.process_item({key: value})
            documents.append(doc)
        
        return documents

class EmbeddingManager:
    """向量存储管理类"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def load_and_embed_documents(self, loader: BaseDocumentLoader, file_path: str):
        """加载文档并存入向量数据库"""
        documents = loader.load_documents(file_path)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
        return len(documents)

# 添加新的函数以支持外部调用
def process_files(collection_name: str, file_paths: List[str], parse_mode: str = "subsection", db_path: str = "./chroma_langchain_db"):
    """处理多个文件并将其嵌入到指定的向量数据库中"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
    )
    
    embedding_manager = EmbeddingManager(vector_store)
    
    for file_path in file_paths:
        if file_path.endswith('.md'):
            loader = TextbookMarkdownLoader(parse_mode=parse_mode)
            num_sections = embedding_manager.load_and_embed_documents(loader, file_path)
            print(f"已加载 {num_sections} 个课本章节来自: {file_path}")
        elif file_path.endswith('.json'):
            loader = ImageDescriptionLoader()
            num_images = embedding_manager.load_and_embed_documents(loader, file_path)
            print(f"已加载 {num_images} 个图片描述来自: {file_path}")
        else:
            print(f"不支持的文件类型: {file_path}")

# 使用示例
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='文档加载器')
    
    # 仅测试模式需要的参数
    parser.add_argument('--test_mode', action='store_true', help='是否启用测试模式')
    
    # 仅正式模式需要的参数
    parser.add_argument('--collection_name', type=str, help='向量数据库名称')
    parser.add_argument('--db_path', type=str, default='./chroma_langchain_db', help='向量数据库创建路径')

    # 测试模式特有的参数
    parser.add_argument('--file_type', type=str, choices=['markdown', 'json'], help='文件类型：markdown-课本文本, json-图片描述')
    parser.add_argument('--output_dir', type=str, default='debug_output', help='测试模式输出目录')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--markdown_path', type=str, help='Markdown 文件路径')
    group.add_argument('--json_path', type=str, help='图片描述 JSON 文件路径')
    parser.add_argument('--parse_mode', type=str, default='subsection', choices=['none', 'chapter', 'section', 'subsection'], help='Markdown解析模式')
    
    args = parser.parse_args()

    if args.test_mode:
        # 测试模式逻辑
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args.file_type == 'markdown':
            input_path = args.markdown_path
            input_filename = os.path.splitext(os.path.basename(input_path))[0]
            debug_output_path = os.path.join(
                args.output_dir, 
                f'textbook_debug_{input_filename}_{timestamp}.json'
            )
            loader = TextbookMarkdownLoader(
                debug_mode=True, 
                debug_output_path=debug_output_path,
                parse_mode=args.parse_mode
            )
            print(f"测试模式：开始处理 Markdown 文件: {input_path}")
            
        elif args.file_type == 'json':
            input_path = args.json_path
            input_filename = os.path.splitext(os.path.basename(input_path))[0]
            debug_output_path = os.path.join(
                args.output_dir, 
                f'images_debug_{input_filename}_{timestamp}.json'
            )
            loader = ImageDescriptionLoader()
            print(f"测试模式：开始处理图片描述 JSON 文件: {input_path}")
        
        documents = loader.load_documents(input_path)
        
        # 将结果写入JSON文件
        debug_output = [{
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in documents]
        
        with open(debug_output_path, 'w', encoding='utf-8') as f:
            json.dump(debug_output, f, ensure_ascii=False, indent=2)
            
        print(f"处理完成！文档详细信息已保存至: {debug_output_path}")
        
    else:
        # 正式模式逻辑
        # 预定义参数字典
        predefined_params = {
            "collection_name": "cell_biology_collection_5th",  # 向量数据库名称
            "file_paths": ["/root/RAG-test/CrossModalRetrieval-RAG/assets/细胞生物学（5）.md","/root/RAG-test/CrossModalRetrieval-RAG/assets/6 细胞生物学（5）_figures_description.json"],  # 文件路径列表
            "parse_mode": "subsection",   # 解析模式
            "db_path": "/root/RAG-test/CrossModalRetrieval-RAG/assets/chroma_langchain_db"     # 向量数据库路径
        }
        
        # 如果字典中有值就使用字典值,否则使用命令行参数
        collection_name = predefined_params["collection_name"] or args.collection_name
        file_paths = predefined_params["file_paths"] or [args.markdown_path]
        parse_mode = predefined_params["parse_mode"] or args.parse_mode
        db_path = predefined_params["db_path"] or args.db_path
        
        # 正式处理逻辑
        process_files(
            collection_name=collection_name,
            file_paths=file_paths, 
            parse_mode=parse_mode,
            db_path=db_path
        )