from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import uuid

from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("./data/faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

prompt_template = PromptTemplate.from_template("""
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

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Streamlit app
st.title("GIGI")

# 세션 ID가 설정되지 않았다면 새로운 UUID를 생성
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
# Handling the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# 초기 메세지 출력
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "안녕하세요. 무엇을 도와드릴까요?"
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("궁금한 것이 있으면 물어보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Execute the RAG chain with the given user input
        response = rag_with_history.invoke(
            {"question": prompt},
            config={"configurable": {"session_id": st.session_state["session_id"]}}
        )

        # Displaying the response in real-time
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
