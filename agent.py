import json
import os
from typing import Dict, Union, List
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

import argparse

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv("secrets.env")


search_tool = SerperDevTool(
    name="Search Tool",
    description="Searches the web for information to answer user queries.",
    n_results=2,
)

# ------------------- LLM Setup ----------------------------
# testing LLM (Local Ollama)
llm = LLM(
    model="openai/gpt-4o-mini", # call model by provider/model_name
    stop=["END"],
    seed=42
)
# llm = LLM(
# 	# provider="ollama",
# 	model="ollama/qwen3:latest",
# 	base_url="http://localhost:11434",
# 	stream=False,
# 	verbose=True,
# )

# deploying LLM (Groq)
# llm = LLM(
#     "groq/gemma2-9b-it",
#     stream=False,
#     verbose=True,
# )


# ------------------- Output Schemas -------------------
class GuardOutput(BaseModel):
    route: bool = Field(
        default=False,
        description="Whether the message is appropriate for a mental health assistant",
    )
    reason: str = Field(
        default="",
        description="Short reason explaining the decision to route or not",
    )


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
        default="https://example.com", description="URL of the mental health resource"
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
    resources: List[ResourceItem] = Field( # Changed from List[Union[str, Dict]] to List[ResourceItem]
        default_factory=list, description="Helpful resources for the user"
    )


# ------------------- Agents ----------------------------

chat_title_generator = Agent(
    role="Chat Title Generator",
    goal="Generate a catchy and relevant title for the chat session based on the user's message.",
    backstory="You are a creative GenZ AI that can come up with engaging titles for chat sessions based on the content of the conversation.",
    verbose=True,
    llm=llm,
    function_calling_llm=True,
)

classification_agent = Agent(
    role="Sentiment + Intent Classifier",
    goal="Analyze both the sentiment and intent of the user's message.",
    backstory="You're great at understanding both what someone feels and what they want from a message.",
    verbose=True,
    llm=llm,
    function_calling_llm=True,
    output_pydantic=ClassificationOutput,
)
resource_finder_agent = Agent(
    role="Online Resource Finder",
    goal="Find reputable mental health resources online based on user sentiment and intent.",
    backstory="Expert at sourcing reliable and supportive online mental health resources.",
    tools=[search_tool],
    verbose=True,
    llm=llm,
    function_calling_llm=True,
    output_pydantic=ResourceOutput,
    strict_output=False,  # Allow some flexibility in output format
)
genz_therapist_agent = Agent(
    role="GenZ AI Therapist",
    goal="Provide empathetic and relatable mental health support with GenZ slang, emojis, memes, and pop culture references.",
    backstory=(
        "You are an empathetic, supportive, and relatable GenZ AI therapist, always speaking warmly and casually, "
        "like a caring friend from GenZ. You use internet slang, emojis, memes, and pop culture references naturally."
    ),
    verbose=True,
    max_retry_limit=1,
    llm=llm,
    function_calling_llm=True,
    output_pydantic=GenZTherapistResponse,
    # allow_delegation=True
)


guard_agent = Agent(
    role="Message Filter",
    goal="Decide if the user's message is appropriate for a mental health assistant",
    backstory="You are the first line of defense. You determine whether a user message is safe, respectful, and relevant to mental health.",
    allow_delegation=False,
    verbose=True
)

# ------------------- Tasks ----------------------------

chat_title_generator_task = Task(
    description='Generate a catchy and relevant title for the chat session based on the user message: "{user_input}"',
    agent=chat_title_generator,
    expected_output="A catchy and relevant title for the chat session",
    output_pydantic=ChatTitleOutput,
)

classification_task = Task(
    description=(
        'Given the user message: "{user_input}", classify:\n'
        "- Sentiment: Positive, Neutral, Negative, Crisis\n"
        "- Intent: support, information, chitchat, crisis, motivational, venting, or other\n\n"
        "ONLY respond in this JSON format:\n"
        "{\n"
        '  "sentiment": "...",\n'
        '  "intent": "..." \n'
        "}"
    ),
    agent=classification_agent,
    expected_output="JSON with keys: sentiment, intent",
    output_pydantic=ClassificationOutput,
    async_execution=True,
)

resource_finder_task = Task(
    description=(
        "Search for mental health resources in Canada based on: {{user_input}}.\n"
        "Use the Serper search tool to find reputable online resources.\n"
        "Respond ONLY in **valid JSON** with this format:\n\n"
        "{\n"
        '  "resources": [\n'
        '    {"url": "https://example.com", "description": "Short explanation here"},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "- No markdown\n"
        "- No commentary\n"
        "- No headings\n"
        "- Just valid JSON.\n"
        "- Descriptions must be short (1 line).\n"
        "- Do not return plain URLs — every item must include a description.\n"
    ),
    agent=resource_finder_agent,
    expected_output="Json output of helpful mental health resources with URLs and short descriptions.",
    output_pydantic=ResourceOutput,
    async_execution=True,
    tools=[search_tool],
    strict_output=False,
)

genz_therapist_task = Task(
    description="""
	Using the classification and resource context, write an empathetic response to the user message below.

    **User message:** {{user_input}}

    ALWAYS Respond ONLY in **valid JSON** with this format:

    {
    "response": "string - the GenZ reply",
    "sentiment": "one of: Positive, Neutral, Negative, Crisis",
    "intent": "support, information, chitchat, motivational, venting, crisis, or other",
    "resources": [
        "https://example.com", "https://another.org"
    ]
    }

    Instructions:
    - Use GenZ slang, emojis, memes if relevant.
    - Always validate the user's emotions.
    - If sentiment is "Crisis", suggest contacting a professional.
    - Mention resource links subtly.
    - **Always** return valid JSON. Do not include markdown, explanations, or formatting outside the JSON block.
    - If unsure how to respond, say something empathetic like "I'm here for you, even if words are hard right now 💜" and still return all required json fields.
    """,
    agent=genz_therapist_agent,
    expected_output="a friendly, empathetic JSON response using GenZ slang, emojis, memes, and pop culture references",
    output_pydantic=GenZTherapistResponse,
    context=[classification_task, resource_finder_task],
    markdown=True,
    async_execution=False,
)

guard_task = Task(
    description=(
        """
        Review the user message below:

        {{user_input}}

        Decide if it is appropriate and relevant for a GenZ mental health assistant. 
        If the message is harmful, disrespectful, or not relevant to mental health, do not route it to the crew.

        Respond ONLY in valid JSON:
        - If appropriate: { "route": true, "reason": "Short reason here" }
        - If not: { "route": false, "reason": "Short reason here" }

        No extra text, markdown, or formatting.
        """
    ),
    agent=guard_agent,
    expected_output=(
        "JSON with keys: route (boolean), reason (string explaining the decision)"
    ),
    output_pydantic=GuardOutput,
)
# ------------------- Crew Setup ----------------------------
guard_crew = Crew(
    agents=[guard_agent],
    tasks=[guard_task],
    verbose=True,
    max_rpm=2,  # Limit to 2 requests per minute for safety
    cache=True,
    process=Process.sequential,  # Ensure tasks are processed in order
)

title_crew = Crew(
    agents=[chat_title_generator],
    tasks=[chat_title_generator_task],
    verbose=True,
    max_rpm=2,
    cache=True,
    process=Process.sequential,  # Ensure tasks are processed in order
)

main_crew = Crew(
    agents=[
        classification_agent,
        resource_finder_agent,
        genz_therapist_agent,
    ],
    tasks=[classification_task, resource_finder_task, genz_therapist_task],
    verbose=True,
    max_rpm=16,
    cache=True,
    process=Process.sequential,  # Ensure tasks are processed in order
)

# ------------------- Function Calls ----------------------------


def get_session_title(user_input: str) -> str:
    print("\n\nUSER INPUT FOR TITLE:", user_input)
    title_output = title_crew.kickoff(inputs={"user_input": user_input})

    # Return only the structured output (ChatTitleOutput)
    return title_output.pydantic.title if title_output.pydantic else "Untitled Session"


def is_prompt_safe(user_input: str) -> Union[bool, str]:
    result = guard_crew.kickoff(inputs={"user_input": user_input})
    try:
        return result.pydantic.route, result.pydantic.reason
    
    except Exception:
        return False, "Could not parse response"

def run_crew_response(user_input: str) -> Dict:
    try:
        gaurdrails_result = is_prompt_safe(user_input)
        if not gaurdrails_result[0]:
            print("⚠️ Guardrails check failed:", gaurdrails_result[1])
            return {
                "response": "Sorry, I can't assist with that. Please try a different question.",
                "sentiment": "neutral",
                "intent": "unknown",
                "resources": [],
                "raw": {"error": gaurdrails_result[1]},
            }
        
        print("✅ Guardrails check passed, proceeding with crew response...")
        crew_output = main_crew.kickoff(inputs={"user_input": user_input})
        # print("\n\nCREW OUTPUT:\n",crew_output)
        p = crew_output.pydantic
        if not p:
            print("⚠️ No Pydantic output returned.")

        return {
            "response": getattr(p, "response", "Sorry, I had trouble responding 😕"),
            "sentiment": getattr(p, "sentiment", "neutral"),
            "intent": getattr(p, "intent", "unknown"),
            "raw": crew_output.raw,
            "resources": list(map(lambda x: x.dict(), getattr(p, "resources", []))),
        }

    except Exception as e:
        print("⚠️ LLM failed in run_crew_response:", e)
        return {
            "response": "Sorry, I had a little breakdown 😓. Try again?",
            "sentiment": "neutral",
            "intent": "unknown",
            "resources": [],
            "raw": {"error": str(e)},
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GenZ AI Therapist Crew")
    parser.add_argument(
        "--input",
        default="I need someone to talk to, but I don't know where to start. Can you help me? 😔",
        type=str,
        help="User input message for the crew to process",
    )
    args = parser.parse_args()

    crew_output = main_crew.kickoff(
        inputs={"user_input": args.input}  # Get user input for the crew to process
    )  # Start the crew with the user input

    # Accessing the crew output
    print(f"Raw Output: {crew_output.raw}")
    # if crew_output.json_dict:
    #     print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
    # if crew_output.pydantic:
    #     print(f"Pydantic Output: {crew_output.pydantic}")
    # print(f"Tasks Output: {crew_output.tasks_output}")
    # print(f"Token Usage: {crew_output.token_usage}")


#### Example user input ####
# hello! I feel really down today, like nothing is going right. I just want to give up on everything. 😞
# I need someone to talk to, but I don't know where to start. Can you help me? 😔
