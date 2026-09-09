"""
大模型客户端 —— 统一封装 ollama OpenAI SiliconFlow 等大模型客户端
"""

import logging
from config.settings import settings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.model_name = settings.LLM_PROVIDER
        self.llm = self._create_llm()
        logger.info(f"已初始化大模型客户端: {self.model_name}")

    def _create_llm(self):
        if self.model_name == "ollama":
            return ChatOllama(
                model=settings.OLLAMA_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.1,
                verbose=True,
            )
        elif self.model_name == "openai":
            return ChatOpenAI(
                model=settings.OPENAI_MODEL_NAME,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                temperature=0.1,
                verbose=True,
            )
        elif self.model_name == "siliconflow":
            return ChatOpenAI(
                model=settings.SILICONFLOW_MODEL_NAME,
                api_key=settings.SILICONFLOW_API_KEY,
                base_url=settings.SILICONFLOW_BASE_URL,
                temperature=0.1,
                verbose=True,
            )
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

    def get_llm(self):
        return self.llm

    def change_llm_client(self, model_name: str, **kwargs):
        self.model_name = model_name
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self._create_llm()
        logger.info(f"已切换大模型客户端: {self.model_name}")


# 实例化大模型客户端
_llm_client = None


def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client.get_llm()
