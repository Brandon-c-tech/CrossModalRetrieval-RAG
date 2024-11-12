import os
from datetime import datetime
from collections import defaultdict
import shutil  # 新增导入

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

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

# 对textbook来源进行相似性搜索，返回得分最高的4个结果，结果以list形式返回，list里是tuple
results_textbook = vector_store.similarity_search_with_score(
    query,
    k=4,
    filter={"source": "textbook"}
)

# 创建一个字典来存储合并后的metadata
merged_metadata = defaultdict(lambda: {"sections": defaultdict(set)})

page_contents = []
page_contents_string = ""

# 处理来自textbook的结果
for res, score in results_textbook:
    page_content = res.page_content
    chapter = res.metadata.get("chapter")
    section = res.metadata.get("section")
    subsection = res.metadata.get("subsection")

    page_contents.append(page_content)

    page_contents_string += (
        f"章节: {chapter}\n"
        f"节: {section}\n"
        f"小节: {subsection}\n"
        f"{page_content}\n\n"
    )
    
    # 合并metadata
    if chapter and section:
        merged_metadata[chapter]["sections"][section].add(subsection)

# 显示引用来源。打印合并后的结果，处理空值情况
references_content = "<h3>引用来源</h3><ul>"  # 初始化引用来源的 HTML 内容
for chapter, sections in merged_metadata.items():
    print(chapter)  # 打印章节
    references_content += f"<li>{chapter}<ul>"  # 添加章节到 HTML
    for section, subsections in sections["sections"].items():
        if section:  # 确保section不为空
            print(f"  {section}")  # 打印节
            references_content += f"<li>{section}<ul>"  # 添加节到 HTML
            for subsection in subsections:
                if subsection:  # 确保subsection不为空
                    print(f"    {subsection}")  # 打印小节
                    references_content += f"<li>{subsection}</li>"  # 添加小节到 HTML
            references_content += "</ul></li>"  # 结束节的列表
    references_content += "</ul></li>"  # 结束章节的列表
references_content += "</ul>"  # 结束引用来源的列表

# 对image来源进行相似性搜索，返回得分最高的2个结果，结果以list形式返回，list里是tuple
results_image = vector_store.similarity_search_with_score(
    query,
    k=2,
    filter={"source": "image"}
)

# 处理来自图片的结果
image_descriptions = {}  # 用于存储图片描述的字典
image_name_map = {}  # 用于存储image到image_name的映射
image_counter = 1  # 图片计数器

for res, score in results_image:
    image_content = res.page_content
    image_name = res.metadata.get("image_name")
    
    # 构建描述字典
    image_key = f'image{image_counter}'
    image_descriptions[image_key] = image_content
    image_name_map[image_key] = image_name  # 存储映射关系
    image_counter += 1

# 现在 image_descriptions 是 {'image1':"图片描述1"，'image2':"图片描述2"}
# image_name_map 是 imagei 到 image name 的映射

print(image_descriptions)
print("搜索完成")

prompt_template = f"""
    你是一位专业的课程答疑助手。请根据以下信息回答学生的问题。请根据下列信息回答学生的问题，回答长度在300-800字左右。确保你的回答准确、全面，并与课程水平相符。
    注意，请无视参考资料中的图xx-xx的描述，这些图片无法被读取。请根据图片描述，用{{image数字}}格式表示，也就是图片描述词典的键去插入合适的图片在答案中。
    插入图片时，需要单独起一行成一段。不要在文字中。

    问题: {query}

    课本文字参考信息: {page_contents_string}

    图片描述词典: {image_descriptions}
    """

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": prompt_template}
    ]
)

print(completion.choices[0].message.content)

# 假设 completion.choices[0].message.content 是 AI 返回的回答
response_content = completion.choices[0].message.content

# 替换图片占位符为实际文件名，html语法
for key, image_name in image_name_map.items():
    # 获取图片描述作为alt文本
    alt_text = image_descriptions[key]
    # 替换格式为 {imageX} 的占位符
    response_content = response_content.replace(
        f"{{{key}}}", 
        f'<img src="{image_name}" alt="{alt_text}" class="image">'
    )

print(response_content)

# 输出最终的 HTML 内容
html_content = f"""
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>图文混合回答</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .image {{ max-width: 100%; height: auto; }}
        p {{ margin: 1em 0; }}  /* 添加段落间距 */
        ul {{ margin: 1em 0; }}  /* 添加列表间距 */
    </style>
</head>
<body>
    {response_content}
    {references_content}  <!-- 添加引用来源 -->
</body>
</html>
"""

# 创建带有时间戳的输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/root/RAG-test/CrossModalRetrieval-RAG/assets/{timestamp}"
os.makedirs(output_dir, exist_ok=True)  # 创建目录

# 复制图片到新建的文件夹
for key, image_name in image_name_map.items():
    shutil.copy(f"/root/RAG-test/CrossModalRetrieval-RAG/assets/figures/{image_name}", output_dir)

# 将 html_content 保存为 HTML 文件
with open(f"{output_dir}/response.html", "w", encoding="utf-8") as f:  # 更新文件路径
    f.write(html_content)

print("HTML 文件已生成。")