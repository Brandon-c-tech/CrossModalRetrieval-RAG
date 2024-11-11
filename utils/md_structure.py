import re
import logging
import os
from datetime import datetime

class MarkdownProcessor:
    def __init__(self, input_path, output_dir):
        """
        初始化处理器
        :param input_path: 输入文件的完整路径
        :param output_dir: 输出目录的路径
        """
        self.input_path = input_path
        self.output_dir = output_dir
        
        # 设置日志
        self.setup_logging()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        input_filename = os.path.basename(input_path)
        self.output_path = os.path.join(
            output_dir, 
            f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{input_filename}"
        )

    def setup_logging(self):
        """设置日志配置"""
        log_filename = f"markdown_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.output_dir, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_markdown(self, content):
        """处理markdown内容"""
        self.logger.info("开始处理markdown文件")
        
        lines = content.split('\n')
        processed_lines = []
        skip_section = False
        current_line_number = 0
        
        # 添加目录收集列表
        table_of_contents = []
        
        stats = {
            'removed_images': 0,
            'removed_sections': 0,
            'modified_headers': 0,
            'converted_to_text': 0
        }
        
        for i, line in enumerate(lines):
            current_line_number += 1
            
            # 跳过思考题和参考文献部分
            if line.strip().startswith('# 思考题') or '参考文献' in line:
                skip_section = True
                stats['removed_sections'] += 1
                self.logger.info(f"行 {current_line_number}: 开始跳过章节 \"{line.strip()}\"")
                continue
                
            if skip_section and line.strip().startswith('#'):
                skip_section = False
                self.logger.info(f"行 {current_line_number}: 结束跳过章节")
                
            if skip_section:
                self.logger.debug(f"行 {current_line_number}: 跳过内容 \"{line.strip()}\"")
                continue
                
            # 跳过图片
            if '![' in line:
                stats['removed_images'] += 1
                self.logger.info(f"行 {current_line_number}: 删除图片标签 \"{line.strip()}\"")
                continue
            if i > 0 and '![' in lines[i-1] and not line.strip().startswith('#'):
                self.logger.info(f"行 {current_line_number}: 删除图片说明文字 \"{line.strip()}\"")
                continue
                
            # 处理标题
            if line.strip().startswith('#'):
                original_line = line
                header_content = line.lstrip('#').strip()
                modified = False
                
                # 一级标题: 第X章
                if re.match(r'^第[一二三四五六七八九十]+章', header_content):
                    line = f"# {header_content}"
                    table_of_contents.append((1, header_content))
                    modified = True
                
                # 二级标题: 第X节
                elif re.match(r'^第[一二三四五六七八九十]+节', header_content):
                    line = f"## {header_content}"
                    table_of_contents.append((2, header_content))
                    modified = True
                    
                # 三级标题: 一、二、三、等
                elif re.match(r'^[一二三四五六七八九十]+、', header_content):
                    line = f"### {header_content}"
                    table_of_contents.append((3, header_content))
                    modified = True
                    
                # 四级标题: (一)或（一）
                elif re.match(r'^\([一二三四五六七八九十]+\)|^（[一二三四五六七八九十]+）', header_content):
                    line = f"#### {header_content}"
                    table_of_contents.append((4, header_content))
                    modified = True
                    
                # 五级标题: 1. 2. 等
                elif re.match(r'^\d+\.', header_content):
                    line = f"##### {header_content}"
                    table_of_contents.append((5, header_content))
                    modified = True
                
                # 如果不符合任何标题格式，转换为正文
                if not modified:
                    line = header_content
                    stats['converted_to_text'] += 1
                    self.logger.info(f"行 {current_line_number}: 将不规范标题转换为正文")
                    self.logger.info(f"  原文: {original_line.strip()}")
                    self.logger.info(f"  转换后: {line.strip()}")
            
            processed_lines.append(line)
        
        # 记录统计信息
        self.logger.info("\n处理总结:")
        self.logger.info(f"- 删除图片数量: {stats['removed_images']}")
        self.logger.info(f"- 删除章节数量: {stats['removed_sections']}")
        self.logger.info(f"- 修改标题数量: {stats['modified_headers']}")
        self.logger.info(f"- 转换为正文数量: {stats['converted_to_text']}")
        
        # 生成目录
        self.logger.info("\n文档目录结构:")
        for level, title in table_of_contents:
            indent = "  " * (level - 1)  # 根据层级缩进
            self.logger.info(f"{indent}{title}")
        
        return '\n'.join(processed_lines)

    def process_file(self):
        """处理文件的主函数"""
        try:
            # 读取文件
            self.logger.info(f"开始读取文件: {self.input_path}")
            with open(self.input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 处理内容
            processed_content = self.process_markdown(content)
            
            # 写入新文件
            self.logger.info(f"写入处理后的文件: {self.output_path}")
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            self.logger.info("文件处理完成")
            return True
            
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            return False

def main():
    # 这里可以设置输入文件路径和输出目录
    input_path = "/root/RAG-test/CrossModalRetrieval-RAG/assets/6 细胞生物学（5） _manual_label.md"  # 替换为实际的输入文件路径
    output_dir = "/root/RAG-test/CrossModalRetrieval-RAG/assets"    # 替换为实际的输出目录
    
    processor = MarkdownProcessor(input_path, output_dir)
    processor.process_file()

if __name__ == "__main__":
    main()
