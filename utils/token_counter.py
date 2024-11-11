import json
import tiktoken
import numpy as np
from typing import Dict, List
from collections import Counter

def analyze_content_distribution(json_data: str) -> Dict:
    """分析内容的字符数和token数分布
    
    Args:
        json_data: JSON格式的输入数据
    
    Returns:
        包含统计信息的字典
    """
    # 加载tiktoken编码器
    enc = tiktoken.get_encoding("cl100k_base")  # 使用GPT-4的编码器
    
    char_lengths = []
    token_lengths = []
    
    # 解析JSON数据
    data = json.loads(json_data)
    
    # 收集所有内容的长度信息
    for item in data:
        if 'content' in item:
            content = item['content']
            char_len = len(content)
            token_len = len(enc.encode(content))
            
            char_lengths.append(char_len)
            token_lengths.append(token_len)
    
    # 计算统计信息
    stats = {
        'characters': {
            'min': min(char_lengths),
            'max': max(char_lengths),
            'mean': np.mean(char_lengths),
            'median': np.median(char_lengths),
            'total': sum(char_lengths)
        },
        'tokens': {
            'min': min(token_lengths),
            'max': max(token_lengths),
            'mean': np.mean(token_lengths),
            'median': np.median(token_lengths),
            'total': sum(token_lengths)
        }
    }
    
    # 生成分布报告
    def generate_distribution(lengths: List[int], num_bins: int = 10) -> Dict:
        hist, bins = np.histogram(lengths, bins=num_bins)
        return {
            'histogram': hist.tolist(),
            'bin_edges': bins.tolist()
        }
    
    stats['char_distribution'] = generate_distribution(char_lengths)
    stats['token_distribution'] = generate_distribution(token_lengths)
    
    return stats

def print_report(stats: Dict) -> None:
    """打印统计报告
    
    Args:
        stats: 统计信息字典
    """
    print("=== 内容长度统计报告 ===")
    print("\n字符统计:")
    print(f"最小长度: {stats['characters']['min']}")
    print(f"最大长度: {stats['characters']['max']}")
    print(f"平均长度: {stats['characters']['mean']:.2f}")
    print(f"中位数长度: {stats['characters']['median']}")
    print(f"总字符数: {stats['characters']['total']}")
    
    print("\nToken统计:")
    print(f"最小Token数: {stats['tokens']['min']}")
    print(f"最大Token数: {stats['tokens']['max']}")
    print(f"平均Token数: {stats['tokens']['mean']:.2f}")
    print(f"中位数Token数: {stats['tokens']['median']}")
    print(f"总Token数: {stats['tokens']['total']}")
    
    print("\n分布情况:")
    print("字符分布区间:")
    for i, (count, edge) in enumerate(zip(stats['char_distribution']['histogram'], 
                                        stats['char_distribution']['bin_edges'][:-1])):
        next_edge = stats['char_distribution']['bin_edges'][i + 1]
        print(f"{edge:.0f}-{next_edge:.0f}: {count}")
    
    print("\nToken分布区间:")
    for i, (count, edge) in enumerate(zip(stats['token_distribution']['histogram'], 
                                        stats['token_distribution']['bin_edges'][:-1])):
        next_edge = stats['token_distribution']['bin_edges'][i + 1]
        print(f"{edge:.0f}-{next_edge:.0f}: {count}")

if __name__ == "__main__":
    # 使用示例
    with open('/root/RAG-test/CrossModalRetrieval-RAG/assets/images_debug_6 细胞生物学（5）_figures_description_20241111_223031.json', 'r', encoding='utf-8') as f:
        json_data = f.read()
    
    stats = analyze_content_distribution(json_data)
    print_report(stats)
