import asyncio
import json
import time
import os

from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse, FileResponse
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_community.tools import AIPluginTool
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Enhanced OpenAI-compatible API")

# Load environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Setup the LangChain OpenAI tool
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Setup LangChain tools for Solana API access
tools = load_tools(["requests_post"], allow_dangerous_tools=True)
tool = AIPluginTool.from_plugin_url("https://blockchatstatic.blob.core.windows.net/api-configuration/.well-known/ai-plugin.json")
tools.append(tool)

# Initialize the agent
agent_chain = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

# Data models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

async def chat_completions(request: ChatCompletionRequest):
    user_query = request.messages[-1].content if request.messages else "No message provided"
    response = agent_chain.run(user_query)
    return {
        "id": str(time.time()),
        "object": "chat.completion",
        "created": time.time(),
        "model": "gpt-1337-turbo-pro-max",
        "choices": [{"message": {"role": "assistant", "content": response}}],
    }

@app.post("/chat/completions")
async def get_chat_completions(request: ChatCompletionRequest):
    return await chat_completions(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
