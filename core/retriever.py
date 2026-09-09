"""
检索模块 —— 支持相似度检索 + 可选的重排序
主要功能：
    1. 基于向量相似度进行文档检索
    2. 可选的文档重排序（使用Reranker）
        使用重排序的原因是为了提高检索结果的相关性和准确性。
        这属于对RAG技术的优化，主要就是为了提高检索结果的质量。
        但是会增加一些额外的开销。
"""

import logging
from config.settings import settings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from core.vector_store import get_vector_store_manager

logger = logging.getLogger(__name__)


class CrossEncoderRetriever(BaseDocumentCompressor):
    model_name: str
    top_k: int
    model = None

    def __init__(self, model_name: str, top_k: int = 5):
        super().__init__()
        self.model_name = model_name
        self.top_k = top_k
        # 延迟初始化模型，避免启动超时

    def _initialize_model(self):
        # 使用时，才加载模型
        if self.model_name is None:
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(self.model_name)
                logger.info(f"已初始化CrossEncoderRetriever模型: {self.model_name}")

            except Exception as e:
                logger.error(f"初始化CrossEncoderRetriever模型时出错: {e}")
                self.model = None

    def compress_documents(self, documents: list, query: str) -> list:
        """
        重排序文档：对文档列表按照查询query相关性排序
        """
        if not documents:
            return []
        # 初始化模型
        self._initialize_model()
        # 模型初始化失败，返回原始文档列表
        if self.model is None:
            return documents[: self.top_k]
        try:
            # 构建查询-文档对
            query_pairs = [[query, document.page_content] for document in documents]
            # 计算查询-文档对的分数
            scores = self.model.predict(query_pairs)
            # 按分数排序,从高到低
            query_pairs_sorted = sorted(
                zip(query_pairs, scores), key=lambda x: x[1], reverse=True
            )
            return [document for _, document in query_pairs_sorted[: self.top_k]]

        except Exception as e:
            logger.error(f"重排序文档时出错: {e}")
            return documents[: self.top_k]


class RAGRetriever:
    def __init__(self):
        self.vector_store_manager = get_vector_store_manager()
        self.retriever = self._create_retriever()

    def _create_retriever(self):
        if not settings.USE_RERANKER:
            logger.info("未启用Reranker，将使用默认检索器")
            return None
        try:
            """
            创建Reranker的检索器
            """
            retriever = CrossEncoderRetriever(
                model_name=settings.RERANKER_MODEL_NAME,
                top_k=settings.SEARCH_TOP_K,
            )
            logger.info(f"已创建Reranker的检索器: {retriever.model_name}")
            return retriever
        except Exception as e:
            logger.error(f"创建Reranker的检索器时出错: {e}")
            return None

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_dict: dict = None,
        return_scores: bool = False,
    ) -> list:
        """
        检索文档
        :param query: 查询字符串
        :param top_k: 返回的文档数量
        :param filter_dict: 过滤条件字典，默认None
        :param return_scores: 是否返回文档分数，默认False
        :return: 文档列表
        """
        # 1. 执行向量检索
        docs_with_score = self.vector_store_manager.similarity_search_with_score(
            query, k=top_k, filter_dict=filter_dict
        )
        # 2. 如果启用了Reranker，则进行重排序
        if self.retriever:
            raw_docs = [doc for doc, _ in docs_with_score]
            ranked_docs = self.retriever.compress_documents(raw_docs, query)
            # 重新匹配原分数
            ranked_docs_with_score = [
                (doc, next(score for doc_, score in docs_with_score if doc_ == doc))
                for doc in ranked_docs
            ]
        else:
            ranked_docs_with_score = docs_with_score

        # 3. 返回结果
        if return_scores:
            return ranked_docs_with_score
        else:
            return [doc for doc, _ in ranked_docs_with_score]

    def get_compresstion_retriever(self, search_kwargs: dict):
        """
        获取适配Langchain链的检索器
        """
        search_kwargs = search_kwargs or {"k": settings.SEARCH_TOP_K}
        # 获取基础向量检索器
        base_retriever = self.vector_store_manager._store.as_retriever(
            search_kwargs=search_kwargs
        )
        # 如果启用了Reranker，则创建压缩检索器
        if self.retriever:
            return ContextualCompressionRetriever(
                base_compressor=self.retriever, base_retriever=base_retriever
            )
        else:
            return base_retriever


# 单例实例（全局唯一，避免重复加载）
_rag_retriever_instance = None


def get_rag_retriever() -> RAGRetriever:
    """获取全局唯一的RAG检索器实例"""
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    return _rag_retriever_instance
