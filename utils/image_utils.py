import os
import json
from openai import OpenAI
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time
from datetime import datetime
import base64
from config.config import settings

class ImageDescriptionGenerator:
    def __init__(self, api_key: str, image_dir: str, output_file: str):
        """
        初始化图片描述生成器
        
        Args:
            api_key (str): OpenAI API密钥
            image_dir (str): 图片目录路径
            output_file (str): 输出JSON文件路径
        """
        self.client = OpenAI(api_key=api_key)
        self.image_dir = Path(image_dir)
        self.output_file = output_file
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        
        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志记录"""
        log_file = f'image_description_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_detailed_prompt(self, image_name: str) -> str:
        """
        创建适用于生物学教学插图和示意图的详细描述 prompt
        
        Args:
            image_name (str): 图片文件名
        
        Returns:
            str: 生成的prompt
        """
        return """请详细描述这张科学教学示意图。要求：

1. 描述长度：控制在100-300字之间

2. 描述结构（请按以下顺序组织内容）：
   A. 开篇概述（15-25字）：
      - 说明图示的主要生物学概念/原理
      - 点明图示类型（如截面图、流程图、结构图等）
   
   B. 核心内容（50-150字）：
      - 详细解释图中展示的生物学原理或概念
      - 描述关键组成部分及其关系
      - 说明重要的因果关系或变化过程
      - 解释图中的箭头、标注等视觉元素含义
   
   C. 教学功能（20-50字）：
      - 说明该图在教学中的作用
      - 指出图示帮助理解的关键点
   
   D. 补充说明（如有必要，15-75字）：
      - 相关的生物学应用场景
      - 与其他生物学概念的联系
      - 特殊的注意事项

3. 描述原则：
   - 使用专业准确的生物学术语
   - 保持逻辑性和连贯性
   - 由表及里，由简到繁
   - 注重概念间的关联性
   
4. 语言要求：
   - 使用生物学教材的规范表述
   - 避免过于口语化的表达
   - 必要时使用专业术语
   - 保持客观严谨的语气

5. 特别注意：
   - 不要使用"这是一张..."等开场白
   - 不评价图片的设计质量
   - 如涉及步骤或过程，要清晰标明顺序
   - 如有数值或单位，需准确描述
   - 如有图例或备注，要包含在描述中

请根据以上要求，为文件名为 {image_name} 的生物学示意图提供详细描述。确保描述准确、专业、系统，并突出教学价值。"""

    def _get_image_description(self, image_path: str, max_retries: int = 3) -> Optional[str]:
        """
        使用 gpt 4o mini 生成图片描述
        
        Args:
            image_path (str): 图片路径
            max_retries (int): 最大重试次数
        
        Returns:
            Optional[str]: 生成的描述或None（如果失败）
        """
        for attempt in range(max_retries):
            try:
                with open(image_path, "rb") as image_file:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self._create_detailed_prompt(Path(image_path).name)
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode()}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=500
                    )
                    return response.choices[0].message.content
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {image_path}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                continue
        return None

    def _get_image_files(self) -> List[Path]:
        """
        获取目录中的所有支持的图片文件
        
        Returns:
            List[Path]: 图片文件路径列表
        """
        image_files = []
        for file_path in self.image_dir.rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        return image_files

    def _load_existing_descriptions(self) -> Dict[str, str]:
        """
        加载已存在的描述文件
        
        Returns:
            Dict[str, str]: 现有的图片描述字典
        """
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning(f"Error reading {self.output_file}, starting fresh")
        return {}

    def _save_descriptions(self, descriptions: Dict[str, str]):
        """
        保存描述到JSON文件
        
        Args:
            descriptions (Dict[str, str]): 图片描述字典
        """
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=4)

    def generate_descriptions(self):
        """
        为所有图片生成描述并保存
        """
        self.logger.info("Starting description generation process")
        
        # 加载现有描述
        descriptions = self._load_existing_descriptions()
        image_files = self._get_image_files()
        
        self.logger.info(f"Found {len(image_files)} image files")
        
        # 处理每个图片
        for image_path in image_files:
            relative_path = str(image_path.relative_to(self.image_dir))
            
            # 跳过已处理的图片
            if relative_path in descriptions:
                self.logger.info(f"Skipping already processed image: {relative_path}")
                continue
                
            self.logger.info(f"Processing image: {relative_path}")
            
            # 获取描述
            description = self._get_image_description(str(image_path))
            
            if description:
                descriptions[relative_path] = description
                self.logger.info(f"Successfully generated description for {relative_path}")
                
                # 定期保存结果
                self._save_descriptions(descriptions)
            else:
                self.logger.error(f"Failed to generate description for {relative_path}")
        
        # 最终保存
        self._save_descriptions(descriptions)
        self.logger.info("Description generation process completed")

def main():
    """
    主函数
    """
    # 配置参数
    api_key = settings.OPENAI_API_KEY
    image_dir = ""
    output_file = "image_descriptions.json"
    
    # 创建生成器实例
    generator = ImageDescriptionGenerator(api_key, image_dir, output_file)
    
    # 生成描述
    generator.generate_descriptions()

if __name__ == "__main__":
    main()