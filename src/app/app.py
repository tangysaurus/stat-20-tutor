from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from tutor import KnowledgeGraph
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from tools import wolfram

load_dotenv()
NEO_PASSWORD = os.getenv("NEO_PASSWORD")
NEO_URI = os.getenv("NEO_URI")
NEO_USERNAME = os.getenv("NEO_USERNAME")

knowledge_graph = KnowledgeGraph(NEO_URI, NEO_USERNAME, NEO_PASSWORD)

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
llm_with_tools = llm.bind_tools([wolfram])

embeddings = OpenAIEmbeddings()
vector_index = knowledge_graph.create_vector_index(embeddings)
entity_chain = knowledge_graph.create_entity_chain(llm)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods = ["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    reply = knowledge_graph.ask_query(llm_with_tools, vector_index, entity_chain, user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug = True)