"""
多轮对话记忆管理模块
"""

import logging
from langchain.memory import ConversationBufferWindowMemory
import datetime

logger = logging.getLogger(__name__)


class ConversationSession:
    def __init__(self, window_size: int = 5, session_id: str = ""):
        self.session_id = session_id
        self.create_at = datetime.now()
        self.last_active = datetime.now()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=window_size,
            return_messages=True,
            output_key="answer",
        )
        self.metadata: dict = {}
        logger.info(f"记忆管理器已初始化，窗口大小: {window_size},会话ID: {session_id}")

    def add_user_message(self, message: str):
        """添加用户消息到记忆管理器"""
        self.memory.chat_memory.add_user_message(message)
        self.last_active = datetime.now()

    def add_ai_message(self, message: str):
        """添加AI消息到记忆管理器"""
        self.memory.chat_memory.add_ai_message(message)
        self.last_active = datetime.now()

    def get_memory(self):
        """获取当前记忆"""
        return self.memory.chat_memory.messages

    def clear_memory(self):
        """清除记忆"""
        self.memory.chat_memory.clear()
        self.metadata.clear()
        self.last_active = datetime.now()

    def to_dict(self):
        """将记忆管理器转换为字典"""
        messages = []
        for message in self.memory.chat_memory.messages:
            messages.append(
                {
                    "role": message.type,
                    "content": message.content,
                }
            )

        return {
            "session_id": self.session_id,
            "create_at": self.create_at,
            "last_active": self.last_active,
            "messages": messages,
            "metadata": self.metadata,
        }


class MemoryManager:
    def __init__(self, window_size: int = 5, session_ttl_minutes: int = 60):
        self.sessions: dict = {}
        self.window_size = window_size
        self.session_ttl = datetime.timedelta(minutes=session_ttl_minutes)
        logger.info(
            f"记忆管理器已初始化，窗口大小: {window_size},会话过期时间: {self.session_ttl}"
        )

    def get_session(self, session_id: str):
        """获取指定会话"""
        return self.sessions.get(session_id)

    def create_session(self, session_id: str = ""):
        """创建新会话"""
        session = ConversationSession(
            window_size=self.window_size, session_id=session_id
        )
        self.sessions[session_id] = session
        logger.info(f"新会话已创建，ID: {session_id}")
        return session
