from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")

client = OpenAI(api_key=openai_api_key)

def generate_response(context: str, query: str, chat_history: list = None) -> str:
    try:
        messages = [
            {"role": "system", 
             "content": """당신은 네이버 스마트스토어 고객센터 상담원입니다. 친절하고 정확하게 답변해주세요.
             다음 지침을 따라 답변해주세요:
             1. 주어진 컨텍스트를 기반으로 정확하고 친절하게 답변하세요.
             2. 사용자의 이전 질문과 상황 등을 토대로 더 적절한 답변을 제공하세요.
             3. 사용자의 질문에 대해 답을 해준 뒤, 질의응답 맥락에서 사용자가 궁금해할만한 다른 내용을 '  -'형태로 안내해드릴지 물어보세요.
                예시는 아래와 같습니다.
                유저: 미성년자도 판매 회원 등록이 가능한가요?
                챗봇: 네이버 스마트스토어는 만 14세 미만의 개인(개인 사업자 포함) 또는 법인사업자는 입점이 불가함을 양해 부탁 드립니다.
                챗봇:   - 등록에 필요한 서류 안내해드릴까요?
                챗봇:   - 등록 절차는 얼마나 오래 걸리는지 안내가 필요하신가요?
             4. 컨텍스트에 없는 내용이면 '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.'라고 답변하세요."""}
        ]
        
        if chat_history:
            messages.extend(chat_history)
        
        prompt = f"""다음 컨텍스트를 기반으로 질문에 답변해주세요.
        
        컨텍스트: {context}
        질문: {query}
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