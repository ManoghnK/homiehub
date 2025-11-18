from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    app_name: str = "Recommendation Service"
    debug: bool = True

    gcloud_json: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
