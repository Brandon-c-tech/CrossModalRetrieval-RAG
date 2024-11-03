from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "your-api-key"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    CHAT_MODEL: str = "gpt-4o-mini"
    VISION_MODEL: str = "gpt-4o-mini"
    
    # 向量数据库设置
    VECTORDB_DIR: str = "vectordb"
    TEXT_COLLECTION_NAME: str = "text_vectors"
    IMAGE_COLLECTION_NAME: str = "image_vectors"
    
    # 检索设置
    TOP_K_TEXT: int = 3
    TOP_K_IMAGES: int = 2
    
    class Config:
        env_file = ".env"

settings = Settings()
