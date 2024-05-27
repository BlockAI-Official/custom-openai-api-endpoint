import time
import os
import wandb

from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import AIPluginTool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain import hub

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_community.callbacks import get_openai_callback

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List
import tiktoken




# Load environment variables from .env file
load_dotenv()

# Access the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
LANGCHAIN_WANDB_TRACING = os.getenv('LANGCHAIN_WANDB_TRACING')
WANDB_PROJECT = os.getenv('WANDB_PROJECT')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

wandb.login(key=WANDB_API_KEY)

app = FastAPI(title="Enhanced OpenAI-compatible API")

### Tools

## Solanalabs API

URL = "https://blockchatstatic.blob.core.windows.net/api-configuration"
tools = load_tools(["requests_post"], allow_dangerous_tools=True)
solanalabs_tool = AIPluginTool.from_plugin_url(URL + "/.well-known/ai-plugin.json")

## Websearch
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

### Memory

## Long term memory

embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

urls = [
    "https://docs.kamino.finance/",
    "https://docs.kamino.finance/kamino-lend-litepaper",
    "https://docs.kamino.finance/products/overview",
    "https://docs.kamino.finance/products/multiply",
    "https://docs.kamino.finance/products/multiply/how-to",
    "https://docs.kamino.finance/products/multiply/how-to/open-a-position",
    "https://docs.kamino.finance/products/multiply/how-to/manage-a-position",
    "https://docs.kamino.finance/products/multiply/how-to/manage-risk",
    "https://docs.kamino.finance/products/multiply/how-it-works",
    "https://docs.kamino.finance/products/multiply/risks",
    "https://docs.kamino.finance/products/borrow-lend",
    "https://docs.kamino.finance/products/borrow-lend/supplying-assets",
    "https://docs.kamino.finance/products/borrow-lend/ktoken-collateral",
    "https://docs.kamino.finance/products/borrow-lend/borrowing-assets",
    "https://docs.kamino.finance/products/borrow-lend/position-risk-and-liquidations",
    "https://docs.kamino.finance/products/borrow-lend/position-risk-and-liquidations/position-risk",
    "https://docs.kamino.finance/products/borrow-lend/position-risk-and-liquidations/borrow-factors",
    "https://docs.kamino.finance/products/borrow-lend/fees",
    "https://docs.kamino.finance/products/long-short",
    "https://docs.kamino.finance/products/liquidity",
    "https://docs.kamino.finance/kamino-points/overview",
    "https://docs.kamino.finance/kamino-points/overview/rates-and-boosts",
    "https://docs.kamino.finance/kamino-points/overview/seasons",
    "https://docs.kamino.finance/kamino-points/overview/seasons/season-1",
    "https://docs.kamino.finance/kamino-points/overview/seasons/season-2"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = Chroma.from_documents(documents=doc_splits,
                                    embedding=embed_model,
                                    collection_name="local-rag")
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

retriever_tool = create_retriever_tool(
    retriever,
    "documentation_search",
    "Search for information about Kamino Finance. For any questions about Kamino Finance, you must use this tool!",
)

## Short term memory

message_history = ChatMessageHistory()

## Sensory memory

prompt = hub.pull("openai-functions-agent")

### Agent

## All tools together

tools += [solanalabs_tool, tavily_tool, retriever_tool]

## Defining an agent with tools and memory

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Data models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class TokenCostProcess:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.successful_requests = 0

    def sum_prompt_tokens(self, tokens: int):
        self.prompt_tokens += tokens

    def sum_completion_tokens(self, tokens: int):
        self.completion_tokens += tokens

    def sum_successful_requests(self, count: int):
        self.successful_requests += count

    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

class CostCalcAsyncHandler(AsyncCallbackHandler):
    model: str = ""
    token_cost_process: TokenCostProcess

    def __init__(self, model, token_cost_process):
        self.model = model
        self.token_cost_process = token_cost_process

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        encoding = tiktoken.encoding_for_model(self.model)
        if self.token_cost_process is None:
            return
        for prompt in prompts:
            self.token_cost_process.sum_prompt_tokens(len(encoding.encode(prompt)))

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.token_cost_process.sum_completion_tokens(1)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.token_cost_process.sum_successful_requests(1)


async def chat_completions(request: ChatCompletionRequest):
    user_query = request.messages[-1].content if request.messages else "No message provided"

    input = {"input": user_query}
    config = {"configurable": {"session_id": "<foo>"}}

    token_cost_process = TokenCostProcess()
    handler = CostCalcAsyncHandler(model="gpt-3.5-turbo", token_cost_process=token_cost_process)

    # Manually calculate the tokens for the prompt
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_cost_process.sum_prompt_tokens(len(encoding.encode(user_query)))

    response = agent_with_chat_history.invoke(input, config=config)

    # Manually calculate the tokens for the response
    response_content = response["output"]
    token_cost_process.sum_completion_tokens(len(encoding.encode(response_content)))

    total_tokens = token_cost_process.total_tokens()
    prompt_tokens = token_cost_process.prompt_tokens
    completion_tokens = token_cost_process.completion_tokens

    return {
        "id": str(time.time()),
        "object": "chat.completion",
        "created": time.time(),
        "model": "gpt-1337-turbo-pro-max",
        "choices": [{"message": {"role": "assistant", "content": response_content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }

@app.post("/chat/completions")
async def get_chat_completions(request: ChatCompletionRequest):
    return await chat_completions(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if not specified
    uvicorn.run(app, host="0.0.0.0", port=port)
