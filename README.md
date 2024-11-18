# COXWAVE_CHATBOT

네이버 스마트스토어의 FAQ를 기반으로 한 대화형 챗봇 애플리케이션입니다. 
FastAPI를 사용하여 구현되었으며, OpenAI의 임베딩 모델과 GPT를 활용하여 사용자의 질문에 맥락을 고려한 답변을 제공합니다.


## 프로젝트 구조

```
COXWAVE_CHATBOT/
├── app/
│   ├── data/
│   │   └── chroma_db/
│   │   └── final_result.pkl
│   │   └── processed_final_reult.pkl
│   ├── templates/
│   │   └── chat.html
│   ├── __init__.py
│   ├── data_load.py
│   ├── data_preprocess.py
│   ├── embedding.py
│   ├── main.py
│   ├── model.py
│   └── rag.py
├── venv/
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

## 설치 및 실행 방법

1. app 디렉토리 아래 data 디렉토리를 생성한 후, final_result.pkl 파일 업로드


2.Python 가상환경 설정 및 활성화:

Windows의 경우:
```bash
python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux의 경우:
```bash
python -m venv venv
source venv/bin/activate
```

3. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```


4. 루트 디렉토리에 `.env` 파일 생성:
```plaintext
OPENAI_API_KEY = "your openai api key"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
```


5. 애플리케이션 실행:
```bash
python -m app.main
```

6. 웹 브라우저에서 아래 주소로 접속:
```
http://0.0.0.0:8000
```

이제 채팅봇을 사용할 수 있습니다!
