import os
from flask import Flask, jsonify, request
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, ChatAnthropic
import faiss
from agent import AutoGPT

app = Flask(__name__)

@app.route('/research', methods=['POST'])
def do_research():
    
    keyword = request.json.get('keyword', '')
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]
    
    embeddings_model = OpenAIEmbeddings()
    
    embeddings_size = 1536
    index = faiss.IndexFlatL2(embeddings_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    
    agent = AutoGPT.from_llm_and_tools(
        ai_name = "AutoResearch",
        ai_role = "Assistant",
        tools = tools,
        llm = ChatOpenAI(temperature=0),
        memory = vectorstore.as_retriever(),
    )
    agent.chain.verbose = True
    
    result = agent.run([f"write a witty, humorous but concise report about {keyword}", f"save the report in the `report` directory"], limit=4)
    
    return jsonify({'status': 'success', 'result':result})


if __name__ == '__main__':
    app.run(debug=True)