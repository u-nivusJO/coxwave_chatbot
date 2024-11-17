import os
import pickle
import re


def preprocess_text(text: str) -> str:
    """텍스트 전처리 함수"""
    
    # '위 도움말이 도움이 되었나요?'부터 '도움말 닫기'까지 텍스트를 삭제
    text = re.sub(r"위 도움말이 도움이 되었나요\?.*?도움말 닫기", "", text, flags=re.S)
    
    text = text.replace("\ufeff", "") # BOM 제거
    text = text.replace("\xa0", " ") # Non-Breaking Space 제거
    text = text.replace("\xa0", " ") # \xa0 제거
    text = re.sub(r"\xa0\d+", " ", text) # \xa0 뒤에 숫자가 오는 패턴 제거
    text = text.replace("\n", " ") # \n 제거
    text = " ".join(text.split()) # 연속된 공백 제거
    
    return text

def preprocess_data(input_path: str, output_path: str) -> None:
    """FAQ 데이터 전처리 및 저장"""
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    
    processed_data = {
        question: preprocess_text(answer)
        for question, answer in data.items()
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(processed_data, f)

def is_processed_data_available(output_path: str) -> bool:
    """전처리된 데이터 파일 존재 여부 확인"""
    return os.path.exists(output_path)