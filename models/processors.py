import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Coroutine
import asyncio

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import time
@dataclass
class SearchResult:
    answer_text: str
    image_paths: Dict[str, str]  # {image_key: image_path}
    references: str  # 修改为 str 类型，不再使用 Dict

class RAGProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = Chroma(
            collection_name="cell_biology_collection_5th",
            embedding_function=self.embeddings,
            persist_directory="/root/moobius_ccs/resources/chroma_langchain_db",
        )
        self.client = OpenAI()

    async def process_query(self, query: str) -> SearchResult:
        # 并行执行两个搜索
        textbook_search, image_search = await asyncio.gather(
            self._search_textbook(query),
            self._search_images(query)
        )
        
        # 处理搜索结果
        merged_metadata, page_contents_string = self._process_textbook_results(textbook_search)
        image_descriptions, image_paths = self._process_image_results(image_search)
        
        # 生成回答
        answer = self._generate_answer(query, page_contents_string, image_descriptions)
        
        # 格式化引用信息
        formatted_references = self._format_references(merged_metadata)
        
        return SearchResult(
            answer_text=answer,
            image_paths=image_paths,
            references=formatted_references
        )

    async def _search_textbook(self, query: str) -> List[Tuple]:
        return await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query, 
            k=4, 
            filter={"source": "textbook"}
        )

    async def _search_images(self, query: str) -> List[Tuple]:
        return await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query,
            k=2,
            filter={"source": "image"}
        )

    def _process_textbook_results(self, results: List[Tuple]) -> Tuple[dict, str]:
        merged_metadata = defaultdict(lambda: {"sections": defaultdict(set)})
        page_contents_string = ""
        
        for res, score in results:
            page_content = res.page_content
            chapter = res.metadata.get("chapter")
            section = res.metadata.get("section")
            subsection = res.metadata.get("subsection")
            
            page_contents_string += (
                f"章节: {chapter}\n"
                f"节: {section}\n"
                f"小节: {subsection}\n"
                f"{page_content}\n\n"
            )
            
            if chapter and section:
                merged_metadata[chapter]["sections"][section].add(subsection)
        
        return dict(merged_metadata), page_contents_string

    def _process_image_results(self, results: List[Tuple]) -> Tuple[dict, dict]:
        image_descriptions = {}
        image_paths = {}
        
        for i, (res, score) in enumerate(results, 1):
            image_key = f'image{i}'
            image_descriptions[image_key] = res.page_content
            image_paths[image_key] = res.metadata.get("image_name")
        
        return image_descriptions, image_paths

    def _generate_answer(self, query: str, page_contents_string: str, image_descriptions: dict) -> str:
        prompt_template = f"""
        你是一位专业的课程答疑助手。请根据提供的课本内容和图片信息，用中文markdown格式为学生解答问题。

        回答要求：
        1. 回答长度控制在300-800字
        2. 语言要专业准确，符合大学本科的学术水平
        3. 可以引用课本内容，但要用自己的语言重新组织表达
        4. 如果插图描述词典中的图片内容与问题相关，可以在回答中自然地引用和解释
        5. 忽略课本文字中出现的"图xx-xx"的引用，因为这些原书图片无法获取
        6. 回答要有逻辑性和连贯性，避免简单罗列知识点
        7. 如果问题涉及多个方面，要分点回答，确保全面性

        问题: {query}

        课本参考内容: {page_contents_string}

        相关图片描述: {image_descriptions}
        """

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt_template}
            ]
        )
        
        return completion.choices[0].message.content

    def _format_references(self, references: Dict) -> str:
        formatted_str = ""
        for chapter, chapter_data in references.items():
            formatted_str += f"章节：{chapter}\n"
            for section_key, sections in chapter_data.items():
                if section_key == "sections":
                    for section, subsections in sections.items():
                        formatted_str += f"  └─ {section}\n"
                        for subsection in subsections:
                            if subsection:  # 只有当subsection不为空时才添加
                                formatted_str += f"      └─ {subsection}\n"
        return formatted_str

if __name__ == "__main__":
    async def main():
        processor = RAGProcessor()
        
        test_query = "细胞膜的结构"
        print("问题:", test_query)

        start_time = time.time()
        result = await processor.process_query(test_query)
        end_time = time.time()
        
        print("回答:", result.answer_text)
        print("\n图片路径:", result.image_paths)
        print("\n参考内容:")
        print(result.references)  # 直接打印格式化后的字符串

        print(f"处理耗时: {end_time - start_time:.2f} 秒")

    asyncio.run(main())