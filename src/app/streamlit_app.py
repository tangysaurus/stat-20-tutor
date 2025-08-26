import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tutor import KnowledgeGraph
from langchain.embeddings import OpenAIEmbeddings
from tools import wolfram
import os

load_dotenv()
NEO_PASSWORD = os.getenv("NEO_PASSWORD")
NEO_URI = os.getenv("NEO_URI")
NEO_USERNAME = os.getenv("NEO_USERNAME")

knowledge_graph = KnowledgeGraph(NEO_URI, NEO_USERNAME, NEO_PASSWORD)

llm = ChatOpenAI(
    model="o4-mini",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
llm_with_tools = llm.bind_tools([wolfram])

embeddings = OpenAIEmbeddings()
vector_index = knowledge_graph.create_vector_index(embeddings)
entity_chain = knowledge_graph.create_entity_chain(llm)

# app config
st.set_page_config(page_title="Stat 20 Tutor", page_icon="🤖")
st.title("Stat 20 Tutor")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hey pal, it's your favorite Stat 20 Tutor! What questions do you have for me?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Ask anything...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(knowledge_graph.ask_query(llm_with_tools, vector_index, entity_chain, user_query))

    st.session_state.chat_history.append(AIMessage(content=response))