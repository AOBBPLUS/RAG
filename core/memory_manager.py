"""
多轮对话记忆管理模块
"""

import logging
import uuid
from langchain.memory import ConversationBufferWindowMemory
import datetime

logger = logging.getLogger(__name__)


class ConversationSession:
    def __init__(self, window_size: int = 5, session_id: str = ""):
        self.session_id = session_id
        self.window_size = window_size
        self.create_at = datetime.datetime.now()
        self.last_active = datetime.datetime.now()
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
        self.last_active = datetime.datetime.now()

    def add_ai_message(self, message: str):
        """添加AI消息到记忆管理器"""
        self.memory.chat_memory.add_ai_message(message)
        self.last_active = datetime.datetime.now()

    def get_memory(self):
        """获取当前记忆"""
        return self.memory.chat_memory.messages

    def clear_memory(self):
        """清除记忆"""
        self.memory.chat_memory.clear()
        self.metadata.clear()
        self.last_active = datetime.datetime.now()

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

    def create_session(self, session_id: str = "") -> str:
        """创建新会话"""
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id in self.sessions:
            logger.warning(f"会话ID {session_id} 已存在")
            return self.sessions[session_id]
        session = ConversationSession(self.window_size, session_id)
        self.sessions[session_id] = session
        logger.info(f"新会话已创建，ID: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> ConversationSession:
        """获取指定会话"""
        session = self.sessions.get(session_id)
        if session:
            session.last_active = datetime.datetime.now()
        return session

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """获取或创建指定会话"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        new_session_id = self.create_session(session_id)
        session = self.get_session(new_session_id)
        return session

    def add_exchange(self, session_id: str, user_message: str, ai_message: str):
        """添加一轮对话记录"""
        session = self.get_session(session_id)
        session.add_user_message(user_message)
        session.add_ai_message(ai_message)

    def get_chat_history(self, session_id: str) -> list:
        """获取指定会话的对话记录"""
        session = self.get_session(session_id)
        if not session:
            return []
        session.last_active = datetime.datetime.now()
        return session.get_memory()

    def clear_session(self, session_id: str):
        """清空指定会话的记忆"""
        session = self.get_session(session_id)
        if session:
            session.clear_memory()
            logger.info(f"会话 {session_id} 已清空")

    def remove_session(self, session_id: str):
        """移除指定会话"""
        session = self.get_session(session_id)
        if session:
            self.sessions.pop(session_id)
            logger.info(f"会话 {session_id} 已移除")

    def clean_expired_sessions(self):
        """清理过期会话"""
        try:
            now = datetime.datetime.now()
            expired_sessions = [
                session_id
                for session_id, session in self.sessions.items()
                if now - session.last_active > self.session_ttl
            ]
        except Exception as e:
            logger.error(f"清理过期会话时出错: {e}")
            return 0
        for session_id in expired_sessions:
            self.remove_session(session_id)
            logger.info(f"会话 {session_id} 已过期并移除")
        return len(expired_sessions)

    def get_all_sessions(self):
        """获取所有会话"""
        return [session.to_dict() for session in self.sessions.values()]

    def get_default_session(self):
        """获取默认会话"""
        return self.get_or_create_session("default").memory


# 全局单例
_memory_manager_instance = None


def get_memory_manager() -> MemoryManager:
    """获取记忆管理器实例，单例"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance
