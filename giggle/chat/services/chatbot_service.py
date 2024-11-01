from ..models import ChatMessage
from langchain.chat_models import ChatOpenAI
import time
import jwt
from dotenv import load_dotenv
import os
import base64

from rest_framework.exceptions import AuthenticationFailed
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ChatBotService():
    def __init__(self):

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("chat/data/faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

        self.retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

        self.prompt_template = PromptTemplate.from_template("""
        당신은 유학생 아르바이트를 위해 필요한 조건을 알려주는 비서입니다.
        당신은 저장된 정보를 활용하여 사용자의 질문을 분석해서, 정확한 대답을 말해주어야 합니다.

        만약 질문의 답을 알지 못한다면, 모른다고 답해야 합니다.
        모든 답변은 한국어를 사용하여 공식적인 어투로 답해야 합니다.

        관련해서 번호를 물어볼 경우 혹은, 대답하기 애매한 경우에만 1345(국번없이)로 전화해야한다고 알려주세요.

        #Previous Chat History:
        {chat_history}

        #Question:
        {question}

        #Context:
        {context}

        #Answer:"""
                                                            )

        self.llm = ChatOpenAI(model_name="gpt-4o-2024-05-13", temperature=0)

        self.chain = (
                {
                    "context": itemgetter("question") | self.retriever,
                    "question": itemgetter("question"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
        )
    def get_user_history_message(self, request):

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            raise AuthenticationFailed('Authorization header missing or invalid')

        token = auth_header.split()[1]
        user_id = self.decode_jwt(token)

        if not user_id:
            raise AuthenticationFailed('Invalid token')
        # 메시지 목록을 리스트로 변환
        messages = ChatMessage.objects.filter(user_id=user_id).order_by('timestamp')

        if not messages:
            # 초기 메시지 설정
            self.save_message(user_id, "assistant", "안녕하세요. 무엇을 도와드릴까요?")
            messages = ChatMessage.objects.filter(user_id=user_id).order_by('timestamp')

        # 직렬화 가능하도록 변환
        messages_dict = []
        for message in messages:
            message_dict = {
                'user_id': message.user_id,
                'role': message.role,
                'message': message.message,
                'timestamp': message.timestamp
            }
            messages_dict.append(message_dict)

        return messages_dict

    def get_user_history(self, user_id):

        messages = ChatMessage.objects.filter(user_id=user_id).order_by('timestamp')
        user_history = ChatMessageHistory()

        if not messages:
            # 초기 메시지 설정
            initial_message = AIMessage(content="안녕하세요. 무엇을 도와드릴까요?")
            user_history.add_message(initial_message)
            self.save_message(user_id, "assistant", initial_message.content)
        else:
            for message in messages:
                role = message.role
                content = message.message
                if role == "user":
                    user_history.add_message(HumanMessage(content=content))
                else:
                    user_history.add_message(AIMessage(content=content))

        return user_history

    def chat_message(self, request, prompt):

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            raise AuthenticationFailed('Authorization header missing or invalid')

        token = auth_header.split()[1]
        user_id = self.decode_jwt(token)

        if not user_id:
            raise AuthenticationFailed('Invalid token')

        user_history = self.get_user_history(user_id)

        self.save_message(user_id, "user", prompt)

        rag_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda: user_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        # Execute the RAG chain with the given user input
        full_response = ""
        response = rag_with_history.invoke(
            {"question": prompt},
            config={"configurable": {"user_id": user_id}}
        )

        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.02)  # 실시간으로 응답을 전송하는 것처럼 느끼게 함

        self.save_message(user_id, "assistant", full_response.strip())

        return {"content" : full_response.strip()}

    def save_message(self, user_id, role, content):
        ChatMessage.objects.create(user_id=user_id, role=role, message=content)

    def decode_jwt(self, token):
        secret_key = base64.b64decode(os.getenv('SECRET_KEY'))
        if not secret_key:
            raise ValueError("SECRET_KEY is not set in environment variables.")

        try:
            print(token)
            payload = jwt.decode(token, secret_key, algorithms=['HS512'])
            print(payload)
            print(payload.get('aid'))
            print(payload.get('rol'))
            return payload.get('aid')

        except jwt.ExpiredSignatureError:
            print("토큰이 만료되었습니다.")
            return None
        except jwt.InvalidTokenError:
            print("유효하지 않은 토큰입니다.")
            return None
