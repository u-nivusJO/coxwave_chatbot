from app.embedding import collection, embed_and_store_faq_data, embedding_func
from app.model import generate_response
from app.data_load import DataLoader
import logging
import contextlib
from typing import List, Dict, Optional

@contextlib.contextmanager
def suppress_logging():
    try:
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        yield
    finally:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('chromadb').setLevel(logging.INFO)

class RAGSystem:
    def __init__(self):
        self.collection = collection
        self.is_initialized = False
        self.embedding_function = embedding_func
        self.data_loader = DataLoader(
            input_path="./app/data/final_result.pkl",
            output_path="./app/data/processed_final_result.pkl"
        )
        
    def initialize(self):
        """FAQ 데이터 로드 및 임베딩"""
        if self.is_initialized:
            return
            
        try:
            faq_data = self.data_loader.load_data()
                
            with suppress_logging():
                embed_and_store_faq_data(faq_data)
            self.is_initialized = True
            
        except Exception as e:
            raise

    def query(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> tuple[str, List[Dict[str, str]]]:
        """
        사용자 질문에 대한 응답 생성
        
        Args:
            user_query (str): 사용자 질문
            chat_history (list, optional): 이전 대화 기록
            
        Returns:
            tuple[str, list]: (생성된 응답, 업데이트된 대화 기록)
        """
        try:
            if not self.is_initialized:
                self.initialize()
            
            if chat_history is None:
                chat_history = []
                
            with suppress_logging():
                results = self.collection.query(
                    query_texts=[user_query],
                    n_results=3
                )
            
            if results["documents"] and results["documents"][0]:
                context = results["documents"][0][0]
                
                updated_history = chat_history + [{"role": "user", "content": user_query}]
                response = generate_response(context, user_query, updated_history)
                final_history = updated_history + [{"role": "assistant", "content": response}]
                
                return response, final_history
            
            return "적절한 답변을 찾을 수 없습니다.", chat_history
                
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}", chat_history