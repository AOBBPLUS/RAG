"""
RAG链问答模块
"""

import logging

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import prompt, ChatPromptTemplate, MessagesPlaceholder

from config.settings import settings
from core.intent_recognizer import IntentRecognizer
from core.llm_client import get_llm_client
from core.memory_manager import get_memory_manager
from core.retriever import get_rag_retriever

logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "default"


class RAGChain:
    def __init__(self):
        self.session_id = DEFAULT_SESSION_ID
        self.llm = get_llm_client()
        self.intent_recognizer = IntentRecognizer()
        self.memory_manager = get_memory_manager()
        self.retriever = get_rag_retriever()
        self._rag_chain = self._create_rag_chain()
        logger.info("RAG问答链初始化完成")

    def _create_rag_chain(self):
        """
        创建RAG问答链
        """
        # 1.改写用户提示词
        contextualize_q_system_prompt = """
                    你是一个专业的query改写专家。你的任务是根据历史对话，将用户的追问改写成一个独立、完整的query。
                    如果用户的提问已经足够独立，请直接返回原问题，不要添加任何额外内容。
                """
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 创建历史感知检索器
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever.get_compresstion_retriever(search_kwargs={"k": settings.SEARCH_TOP_K}),
            contextualize_q_prompt,
        )
        logger.info("历史感知检索器创建完成")
        qa_system_prompt = """
                   你是一个专业的企业知识助手，请根据以下提供的上下文信息回答用户问题。
                   【重要规则】
                   - 对于与企业知识库相关的问题，优先使用上下文中的内容。
                   - 对于不需要知识库的问题（如数学计算、常识性问题等），可以直接回答。
                   - 如果上下文不足以回答与知识库相关的问题，请明确回复：“根据现有知识库无法回答该问题”。
                   - 回答时请保持专业、客观，避免使用主观观点。
                   - 不要输出任何乱码、特殊符号或不连贯的词语。
                   【上下文信息】
                   {context}
               """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 创建文档合并链
        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt,
        )
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )
        logger.info("RAG问答链创建成功")
        return rag_chain

    def ask(self, question: str, session_id: str = DEFAULT_SESSION_ID) -> dict:
        if not question.strip():
            return {
                "answer": "问题不能为空",
                "intent": "无效输入"
            }
        # 1.识别意图
        intent = self.intent_recognizer.recognize_intent(question)
        logger.info(f"意图识别: {question}")
        # 2. 获取对话历史
        try:
            chat_history = self.memory_manager.get_chat_history(session_id)
            chain_input = {
                "input": question,
                "chat_history": chat_history
            }
            result = self._rag_chain.invoke(chain_input)
            answer = result["answer"]
            source_docs = result.get("context", [])
            sources = []
            for doc in source_docs:
                sources.append({
                    "source": doc.metadata.get("file_name", doc.metadata.get("source", "未知")),
                    "page": doc.metadata.get("page", 1),
                    "content": doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })
            # 3. 更新对话历史
            self.memory_manager.add_exchange(session_id, question, answer)
            return {
                "answer": answer,
                "sources": sources,
                "intent": intent
            }
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            return {"answer": "抱歉，处理问题时出错了", "sources": [], "intent": "系统错误"}


def get_chat_history(self, session_id: str = DEFAULT_SESSION_ID) -> list:
    """获取对话历史"""
    messages = self.memory_manager.get_chat_history(session_id)
    return [
        {"role": "human", "content": msg.content} if isinstance(msg, HumanMessage)
        else {"role": "assistant", "content": msg.content}
        for msg in messages
    ]


def clear_session(self, session_id: str = DEFAULT_SESSION_ID):
    """清空会话历史"""
    self.memory_manager.clear_session(session_id)
    logger.info(f"会话 {session_id} 已清空")


# 全局单例
_rag_chain_instance = None


def get_rag_chain() -> RAGChain:
    """获取RAG问答链单例"""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = RAGChain()
    return _rag_chain_instance
