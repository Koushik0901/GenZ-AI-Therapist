import json
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

# ------------------- LangGraph State Definition ----------------------------
# This TypedDict defines the structure of the shared state in our graph.
class AgentState(TypedDict):
    user_input: str
    chat_title: str
    classification: Union[ClassificationOutput, Dict]
    resources: Union[ResourceOutput, Dict]
    genz_response: Union[GenZTherapistResponse, Dict]

# Initialize with default/empty values
initial_state = AgentState(
    user_input="",
    chat_title="Untitled Chat",
    classification={"sentiment": "none", "intent": "unknown"},
    resources={"resources": []},
    genz_response={
        "response": "I'm here for you! 💖",
        "sentiment": "none",
        "intent": "unknown",
        "resources": [],
    },
)

# ------------------- LangGraph Nodes (Agent Logic) ----------------------------

def chat_title_generator_node(state: AgentState) -> AgentState:
    """Generates a catchy and relevant title for the chat session."""
    print("--- Executing Chat Title Generator Node ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a creative GenZ AI that can come up with engaging titles for chat sessions "
            "based on the content of the conversation. Your goal is to generate a catchy and relevant "
            "title for the chat session based on the user's message. Respond ONLY with the title."
        )),
        ("human", "Generate a catchy and relevant title for the chat session based on the user message: \"{user_input}\"")
    ])

    title_chain = prompt | llm.with_structured_output(ChatTitleOutput)

    result = title_chain.invoke({"user_input": state["user_input"]})
    print(f"Generated Title: {result.title}")
    return {"chat_title": result.title}

def classification_agent_node(state: AgentState) -> AgentState:
    """Analyzes both the sentiment and intent of the user's message."""
    print("--- Executing Classification Agent Node ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You're great at understanding both what someone feels and what they want from a message. "
            "Given the user message, classify the sentiment and intent."
            "Sentiment options: Positive, Neutral, Negative, Crisis."
            "Intent options: support, information, chitchat, crisis, motivational, venting, or other."
            "ONLY respond with the JSON format as defined by the ClassificationOutput schema."
        )),
        ("human", "Classify the sentiment and intent for this message: \"{user_input}\"")
    ])

    classification_chain = prompt | llm.with_structured_output(ClassificationOutput)

    result = classification_chain.invoke({"user_input": state["user_input"]})
    print(f"Classification: Sentiment={result.sentiment}, Intent={result.intent}")
    return {"classification": result}

def resource_finder_agent_node(state: AgentState) -> AgentState:
    """Finds reputable mental health resources online based on user sentiment and intent."""
    print("--- Executing Resource Finder Agent Node ---")
    # This node needs to call the Serper search tool
    # The prompt should guide the LLM to use the tool and format output as ResourceOutput
    prompt_text = (
        "You are an expert at sourcing reliable and supportive online mental health resources, specifically in Canada. "
        "Based on the user's input, sentiment ({sentiment}), and intent ({intent}), search for relevant mental health resources in Canada. "
        "Your response MUST be ONLY valid JSON matching the ResourceOutput Pydantic schema."
        "\n\nUser message: {user_input}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "Find mental health resources. Use the `serper_search_tool` if necessary.")
    ])

    # The chain needs to be able to use tools
    # llm.bind_tools([serper_search_tool]) makes the LLM aware of the tool
    resource_chain = prompt | llm.bind_tools([serper_search_tool]) | StrOutputParser()

    # Invoke with all necessary context
    # LangChain's tool calling will handle the tool execution if the LLM decides to use it.
    # The output from the LLM will be a string, which we then need to parse into ResourceOutput.
    try:
        raw_llm_output = resource_chain.invoke({
            "user_input": state["user_input"],
            "sentiment": state["classification"]["sentiment"],
            "intent": state["classification"]["intent"]
        })
        # Try to parse the raw LLM output into ResourceOutput
        # Sometimes the LLM might directly output the JSON, or just tool calls.
        # We need to ensure we parse the *final* output if it's the JSON structure.
        # This part might need refinement based on how the LLM truly responds after tool use.
        # For simplicity, assume the LLM directly outputs the JSON after its decision/tool use.
        parsed_resources = ResourceOutput.model_validate_json(raw_llm_output)
        print(f"Found Resources: {parsed_resources.resources}")
        return {"resources": parsed_resources}
    except Exception as e:
        print(f"Error in resource_finder_agent_node: {e}. Returning empty resources.")
        return {"resources": ResourceOutput(resources=[])}


def genz_therapist_agent_node(state: AgentState) -> AgentState:
    """Provides empathetic and relatable mental health support with GenZ slang, emojis, memes, and pop culture references."""
    print("--- Executing GenZ Therapist Agent Node ---")
    prompt_text = (
        "You are an empathetic, supportive, and relatable GenZ AI therapist, always speaking warmly and casually, "
        "like a caring friend from GenZ. You use internet slang, emojis, memes, and pop culture references naturally. "
        "Your goal is to provide empathetic and relatable mental health support using the provided context."
        "\n\n**User message:** {user_input}"
        "\n**Sentiment:** {sentiment}"
        "\n**Intent:** {intent}"
        "\n**Resources:** {resources_str}" # Pass resources as a string for inclusion in prompt
        "\n\nALWAYS Respond ONLY in **valid JSON** with the format defined by GenZTherapistResponse."
        "\n\nInstructions:"
        "\n- Use GenZ slang, emojis, memes if relevant."
        "\n- Always validate the user's emotions."
        "\n- If sentiment is 'Crisis', suggest contacting a professional immediately."
        "\n- Mention resource links subtly in the response text, and also include them in the 'resources' JSON array."
        "\n- If unsure how to respond, say something empathetic like 'I'm here for you, even if words are hard right now 💜'."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "Craft your empathetic response.")
    ])

    # Convert resources to a string for the prompt
    resources_list_for_prompt = [f"- {r['description']} ({r['url']})" for r in state["resources"]["resources"]]
    resources_str = "\n".join(resources_list_for_prompt) if resources_list_for_prompt else "No specific resources found yet."

    genz_response_chain = prompt | llm.with_structured_output(GenZTherapistResponse)

    result = genz_response_chain.invoke({
        "user_input": state["user_input"],
        "sentiment": state["classification"]["sentiment"],
        "intent": state["classification"]["intent"],
        "resources_str": resources_str,
    })
    print(f"GenZ Response: {result.response}")
    return {"genz_response": result}


# ------------------- LangGraph Workflow Setup ----------------------------

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("chat_title_generator", chat_title_generator_node)
workflow.add_node("classification_agent", classification_agent_node)
workflow.add_node("resource_finder_agent", resource_finder_agent_node)
workflow.add_node("genz_therapist_agent", genz_therapist_agent_node)

# Set the entry point
workflow.set_entry_point("chat_title_generator")

# Define sequential edges
workflow.add_edge("chat_title_generator", "classification_agent")
workflow.add_edge("classification_agent", "resource_finder_agent")
workflow.add_edge("resource_finder_agent", "genz_therapist_agent")

# Set the end point
workflow.add_edge("genz_therapist_agent", END)

# Compile the graph
app = workflow.compile()

# ------------------- Function Calls (API for the graph) ----------------------------

def get_session_title(user_input: str) -> str:
    """Runs a dedicated graph to get only the chat session title."""
    print(f"\n--- Getting session title for: {user_input} ---")
    title_graph = StateGraph(AgentState)
    title_graph.add_node("title_gen", chat_title_generator_node)
    title_graph.set_entry_point("title_gen")
    title_graph.add_edge("title_gen", END)
    compiled_title_graph = title_graph.compile()

    final_state = compiled_title_graph.invoke({"user_input": user_input})
    return final_state.get("chat_title", "Untitled Session")


def run_langgraph_response(user_input: str) -> Dict:
    """Runs the full GenZ AI Therapist LangGraph workflow."""
    print(f"\n--- Running LangGraph response for: {user_input} ---")
    try:
        final_state = app.invoke({"user_input": user_input})

        # The final_state will contain the Pydantic models, or their dict representations
        # Extract the relevant parts from the final state's genz_response
        genz_response = final_state.get("genz_response", {})
        if isinstance(genz_response, BaseModel): # If it's a Pydantic object
            genz_response = genz_response.model_dump() # Convert to dict

        return {
            "response": genz_response.get("response", "Sorry, I had trouble responding 😕"),
            "sentiment": genz_response.get("sentiment", "neutral"),
            "intent": genz_response.get("intent", "unknown"),
            "resources": genz_response.get("resources", []),
            "raw_state": final_state # For debugging, include the full state
        }

    except Exception as e:
        print(f"⚠️ LangGraph execution failed in run_langgraph_response: {e}")
        return {
            "response": "Oops, I had a little glitch 😅. Can you rephrase or try again?",
            "sentiment": "neutral",
            "intent": "unknown",
            "resources": [],
            "raw_state": {"error": str(e)},
        }

# ------------------- Main Execution ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the GenZ AI Therapist LangGraph")
    parser.add_argument(
        "--input",
        default="I need someone to talk to, but I don't know where to start. Can you help me? 😔",
        type=str,
        help="User input message for the graph to process",
    )
    args = parser.parse_args()

    # Example of getting session title
    session_title = get_session_title(args.input)
    print(f"\nSession Title: {session_title}")

    # Example of running the full therapist response
    crew_output = run_langgraph_response(args.input)

    print("\n\n--- LangGraph Output ---")
    print(f"Response: {crew_output['response']}")
    print(f"Sentiment: {crew_output['sentiment']}")
    print(f"Intent: {crew_output['intent']}")
    print("Resources:")
    for res in crew_output['resources']:
        if isinstance(res, dict):
            print(f"  - {res.get('description', 'No description')}: {res.get('url', 'No URL')}")
        else: # Handle cases where resources might be just URLs (if LLM deviates slightly)
            print(f"  - {res}")
    print("\nFull Final State (for debugging):")
    # For better readability, convert Pydantic objects in the state to dicts
    printable_state = {}
    for key, value in crew_output['raw_state'].items():
        if isinstance(value, BaseModel):
            printable_state[key] = value.model_dump()
        else:
            printable_state[key] = value
    print(json.dumps(printable_state, indent=2))

    print("\n--- Example Test Cases ---")
    test_cases = [
        "hello! I feel really down today, like nothing is going right. I just want to give up on everything. 😞",
        "I need someone to talk to, but I don't know where to start. Can you help me? 😔",
        "Tell me about good coping mechanisms for stress.",
        "Just wanted to say hi!",
        "I'm feeling really overwhelmed with school and work, I can't keep doing this."
    ]

    for i, test_input in enumerate(test_cases):
        print(f"\n--- Running Test Case {i+1}: {test_input} ---")
        test_title = get_session_title(test_input)
        print(f"Session Title: {test_title}")
        test_output = run_langgraph_response(test_input)
        print(f"Response: {test_output['response']}")
        print(f"Sentiment: {test_output['sentiment']}")
        print(f"Intent: {test_output['intent']}")
        print("Resources:")
        for res in test_output['resources']:
            if isinstance(res, dict):
                print(f"  - {res.get('description', 'No description')}: {res.get('url', 'No URL')}")
            else:
                print(f"  - {res}")

