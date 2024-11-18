import os
import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, Optional
import tiktoken
import logging
from dotenv import load_dotenv


def silent_operation(func):
    """로깅을 일시적으로 비활성화하는 데코레이터"""
    def wrapper(*args, **kwargs):
        current_level = logging.getLogger().getEffectiveLevel()
        try:
            logging.getLogger().setLevel(logging.ERROR)
            return func(*args, **kwargs)
        finally:
            logging.getLogger().setLevel(current_level)
    return wrapper

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")

client = chromadb.PersistentClient(path="./app/data/chroma_db")

embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name=embedding_model
)

# 컬렉션 생성
collection = client.get_or_create_collection(
    name="naver_smartstore_faq",
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

def num_tokens_from_string(string: str) -> int:
    """문자열의 토큰 수 계산"""
    encoding = tiktoken.encoding_for_model(embedding_model)
    return len(encoding.encode(string))

def truncate_string_to_tokens(string: str, max_tokens: int = 8000) -> str:
    """최대 토큰 수에 맞게 문자열 자르기"""
    encoding = tiktoken.encoding_for_model(embedding_model)
    tokens = encoding.encode(string)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        string = encoding.decode(tokens)
    return string

@silent_operation
def get_all_existing_ids():
    """기존 컬렉션의 모든 ID 가져오기"""
    try:
        result = collection.get(include=[], backend='rest')
        return set(result['ids'])
    except Exception:
        return set()

def embed_and_store_faq_data(faq_dict: Dict[str, str]) -> Optional[chromadb.Collection]:
   try:
       new_documents = []
       new_metadatas = []
       new_ids = []
              
       for question, answer in faq_dict.items():
           doc_id = question.replace("/", "-")
           
           truncated_answer = truncate_string_to_tokens(answer, max_tokens=8000)
           
           new_documents.append(truncated_answer)
           new_metadatas.append({"question": question})
           new_ids.append(doc_id)
                  
       if new_documents:
           batch_size = 100
           batches_processed = 0
           
           for i in range(0, len(new_documents), batch_size):
               try:
                   batch_end = min(i + batch_size, len(new_documents))
                   collection.add(
                       documents=new_documents[i:batch_end],
                       metadatas=new_metadatas[i:batch_end], 
                       ids=new_ids[i:batch_end]
                   )
                   batches_processed += 1
                   print(f"배치 {batches_processed} 임베딩 완료 ({i+batch_size}/{len(new_documents)})")
               except Exception as e:
                   print(f"배치 {batches_processed} 임베딩 중 오류: {str(e)}")
                   continue
       
       print("embedding done...")
       return collection
       
   except Exception as e:
       print(f"임베딩 중 오류 발생: {str(e)}")
       return None
    
    
