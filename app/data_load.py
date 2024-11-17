import pickle
from typing import List, Dict
from app.data_preprocess import is_processed_data_available, preprocess_data


class DataLoader:
    def __init__(self, input_path: str, output_path: str):
        """
        데이터 로더 초기화
        input_path: 원본 데이터 경로
        output_path: 전처리된 데이터 저장 경로
        """
        self.input_path = input_path
        self.output_path = output_path
    
    def load_data(self) -> List[Dict]:
        """FAQ 데이터 로드 함수"""
        if not is_processed_data_available(self.output_path):
            preprocess_data(self.input_path, self.output_path)
            
        with open(self.output_path, "rb") as f:
            return pickle.load(f)