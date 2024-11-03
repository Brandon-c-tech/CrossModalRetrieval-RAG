import asyncio
from typing import Dict, List
from models.vectorstores import VectorStoreManager
from retriever.dual_retriever import DualRetriever
from utils.image_utils import ImageProcessor
from config.config import settings

class ImageTextRAG:
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.vector_manager.init_vectorstores()
        self.retriever = DualRetriever(
            self.vector_manager.text_vectorstore,
            self.vector_manager.image_vectorstore
        )
        
    def index_content(self, texts: List[str], image_directory: str):
        """索引文本和图片内容"""
        # 索引文本
        self.vector_manager.add_texts(texts)
        
        # 处理和索引图片
        image_descriptions = ImageProcessor.process_directory(image_directory)
        self.vector_manager.add_image_descriptions(
            list(image_descriptions.values()),
            list(image_descriptions.keys())
        )
        
    async def query(self, user_query: str) -> Dict:
        """处理用户查询"""
        # 获取检索结果
        retrieval_results = self.retriever.get_relevant_documents(user_query)
        
        # 重排序结果
        processed_results = self.retriever.rerank_results(user_query, retrieval_results)
        
        return self._format_response(processed_results)
    
    def _format_response(self, processed_results: Dict) -> Dict:
        """格式化最终响应"""
        return {
            "content": processed_results["text_segments"],
            "images": [
                {
                    "path": img_path,
                    "position": pos
                }
                for img_path, pos in zip(
                    processed_results["images"],
                    processed_results["display_order"]
                )
            ]
        }

# 使用示例
async def main():
    # 初始化RAG系统
    rag_system = ImageTextRAG()
    
    # 索引示例内容
    sample_texts = [
        "这是第一段示例文本。",
        "这是第二段示例文本。",
        "这是第三段示例文本。"
    ]
    image_directory = "path/to/your/images"
    
    # 索引内容
    rag_system.index_content(sample_texts, image_directory)
    
    # 测试查询
    query = "你的查询"
    response = await rag_system.query(query)
    print("查询结果:", response)

if __name__ == "__main__":
    asyncio.run(main())