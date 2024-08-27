from ..models import ChatMessage
from langchain.chat_models import ChatOpenAI
import time

from dotenv import load_dotenv
import os

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
        vectorstore = FAISS.load_local("chatbot/data/faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

        self.retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

        self.prompt_template = PromptTemplate.from_template("""
        당신은 유학생 아르바이트를 위해 필요한 조건을 알려주는 비서입니다.
        당신은 저장된 정보를 활용하여 사용자의 질문을 분석해서, 정확한 대답을 말해주어야 합니다.

        만약 질문의 답을 알지 못한다면, 모른다고 답해야 합니다.
        모든 답변은 한국어를 사용하여 공식적인 어투로 답해야 합니다.

        관련해서 번호를 물어볼 경우 1345(국번없이)로 전화해야한다고 알려주세요.

        #Previous Chat History:
        {chat_history}

        #Question:
        {question}

        #Context:
        {context}

        #Answer:"""
                                                            )

        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

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
    def get_user_history_message(self, user_id):
        # 메시지 목록을 리스트로 변환
        messages = list(ChatMessage.objects(user_id=user_id).order_by('timestamp'))

        if not messages:
            # 초기 메시지 설정
            self.save_message(user_id, "assistant", "안녕하세요. 무엇을 도와드릴까요?")
            messages = list(ChatMessage.objects(user_id=user_id).order_by('timestamp'))

        # 직렬화 가능하도록 변환
        messages_dict = []
        for message in messages:
            message_dict = message.to_mongo().to_dict()
            message_dict.pop('_id', None)  # _id 필드를 제거
            messages_dict.append(message_dict)

        return messages_dict


    def get_user_history(self, user_id):
        messages = ChatMessage.objects(user_id=user_id).order_by('timestamp')
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

    def chat_message(self, user_id, prompt):
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

        return full_response.strip()

    def save_message(self, user_id, role, content):
        chat_message = ChatMessage(
            user_id=user_id,
            role=role,
            message=content
        )
        chat_message.save()