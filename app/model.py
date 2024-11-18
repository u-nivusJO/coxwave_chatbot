from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")

client = OpenAI(api_key=openai_api_key)

def generate_response(context: str, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    try:
        messages = [
            {"role": "system", 
             "content": """당신은 네이버 스마트스토어 고객센터 상담원입니다. 친절하고 정확하게 답변해주세요.
             다음 지침을 따라 답변해주세요:
             1. 주어진 컨텍스트를 기반으로 정확하고 친절하게 답변하세요.
             2. 사용자의 이전 질문과 상황 등을 토대로 더 적절한 답변을 제공하세요.
             3. 답변 후에는 반드시 현재 상황에서 추가로 안내해드릴 수 있는 2-3가지 관련 정보를 제안하세요.
                - 각 제안은 반드시 새로운 줄에서 '  - ' 형식으로 시작해야 합니다.
                - "추가 문의 사항이 있으신가요?"와 같은 포괄적인 질문은 하지 마세요.
                - 챗봇이 추가로 안내해줄 수 있는 구체적인 정보나 서비스를 제안해야 합니다.
                    예시:
                    유저: 미성년자도 판매 회원 등록이 가능한가요?
                    챗봇: 네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
                    챗봇:   - 등록에 필요한 서류 안내해드릴까요?
                    챗봇:   - 등록 절차는 얼마나 오래 걸리는지 안내가 필요하신가요?
             4. 컨텍스트에 없는 내용이면 '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.'라고 답변하세요."""}
        ]
        
        if chat_history:
            messages.extend(chat_history[-4:])  # 최근 4개의 메시지만 사용
        
        prompt = f"""다음 컨텍스트를 기반으로 질문에 답변해주세요.
        
        컨텍스트: {context}
        질문: {query}
        
        답변 작성 시 유의사항:
        1. 먼저 질문에 대한 명확한 답변을 제공하세요.
        2. 답변 후에는 하나의 빈 줄을 넣으세요.
        3. 그 다음 줄부터 '  - '로 시작하는 2-3개의 추가 안내 사항을 제안하세요.
           - "~을 안내해드릴까요?", "~에 대해 설명해드릴까요?" 형식으로 작성하세요.
           - 챗봇이 제공할 수 있는 추가 정보나 서비스를 제안하세요.
           - 사용자에게 질문하는 것이 아닌, 추가 안내를 제안하는 형식이어야 합니다.
        
        답변:"""
        
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다."