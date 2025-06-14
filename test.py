import yaml
import os
from typing import Dict, Union, List, TypedDict

# LangChain/LangGraph Imports
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# Pydantic for structured output
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv("secrets.env")

# Ensure API keys are set
# Check if SERPER_API_KEY is set
if not os.getenv("SERPER_API_KEY"):
    print("Warning: SERPER_API_KEY is not set. SerperDevTool will not work.")

# Check if OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY is not set. ChatOpenAI will not work.")

# ------------------- LLM Setup ----------------------------
# Using ChatOpenAI for gpt-4o-mini, which supports function calling natively.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, stop=["END"])

# ------------------- Tools ----------------------------
# Wrap the SerperDevTool as a LangChain tool
search = GoogleSerperAPIWrapper()
serper_search_tool = Tool(
    name="search_tool",
    func=search.run,
    description="A search tool that uses Serper to find information and online resources.",
)

# ------------------- Output Schemas (Re-used from CrewAI) -------------------
class ChatTitleOutput(BaseModel):
    title: str = Field(
        default="Untitled Chat",
        description="Catchy and relevant title for the chat session",
    )

class ClassificationOutput(BaseModel):
    sentiment: str = Field(default="none", description="Sentiment of the user message")
    intent: str = Field(default="unknown", description="Intent behind the user message")

class ResourceItem(BaseModel):
    url: str = Field(
        default="[https://example.com](https://example.com)", description="URL of the mental health resource"
    )
    description: str = Field(
        default="No description provided",
        description="Brief description of the resource",
    )

class ResourceOutput(BaseModel):
    resources: List[ResourceItem] = Field(
        default_factory=list,
        description="List of mental health resources with URLs and descriptions",
    )

class GenZTherapistResponse(BaseModel):
    response: str = Field(
        default="I'm here for you! 💖", description="Empathetic GenZ-styled response"
    )
    sentiment: str = Field(
        default="none", description="The sentiment from classification"
    )
    intent: str = Field(default="unknown", description="The intent from classification")
    resources: List[Union[str, Dict]] = Field(
        default_factory=list, description="Helpful resources for the user"
    )

# ------------------- Load prompts ----------------------------
def load_prompt(file_path: str) -> str:
    """Load a prompt from a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)["prompt"]
    
# Load the main prompt for the GenZ AI Therapist
main_prompt = load_prompt("prompts/genz_therapist_prompt.yaml")



# ------------------- LangGraph Agent Definition ----------------------------

classification_agent = create_react_agent(
    model=llm,
    tools=[],
    response_format=ClassificationOutput,
    prompt=ChatPromptTemplate.from_template(main_prompt)
)