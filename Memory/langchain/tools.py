from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os

# # Google search api tool
# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["SERPAPI_API_KEY"] = "e260964668823249ebafcbcc349cfa56b85fb0e13ffece0461180ba5357baac9"
# llm = OpenAI(temperature=0.9)
# tools = load_tools(["serpapi"], llm=llm)
# agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
# agent.run("who is the Chinese president?")

# BiliBili
from langchain.document_loaders import BiliBiliLoader
loader = BiliBiliLoader(["https://www.bilibili.com/video/BV1xt411o7Xu/"])
loader.load()