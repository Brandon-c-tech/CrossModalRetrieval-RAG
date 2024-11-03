from typing import Dict, List
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from config.config import settings

class ResultProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.CHAT_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    def process_results(self, query: str, results: Dict[str, List[Document]]) -> Dict:
        """处理检索结果，包括结果过滤、合并和排序"""
        # 过滤低相关性结果
        filtered_results = self._filter_results(results)
        
        # 合并和排序结果
        merged_results = self._merge_results(query, filtered_results)
        
        return merged_results
    
    def _filter_results(self, results: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
        """过滤低质量或低相关性的结果"""
        filtered = {
            "text_results": [
                doc for doc in results["text_results"]
                if getattr(doc, "score", 0) > 0.5  # 假设有相似度分数
            ],
            "image_results": [
                doc for doc in results["image_results"]
                if getattr(doc, "score", 0) > 0.5
            ]
        }
        return filtered
    
    def _merge_results(self, query: str, results: Dict[str, List[Document]]) -> Dict:
        """合并文本和图片结果，确定最佳展示顺序"""
        # 这里可以实现更复杂的合并逻辑
        return {
            "text_segments": [doc.page_content for doc in results["text_results"]],
            "images": [doc.metadata["image_path"] for doc in results["image_results"]],
            "display_order": list(range(
                len(results["text_results"]) + len(results["image_results"])
            ))
        }