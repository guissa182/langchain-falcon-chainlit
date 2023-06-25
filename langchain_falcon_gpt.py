import os
import re

from typing import List, Union

from langchain import LLMChain
from langchain.tools import GooglePlacesTool
from langchain.chat_models import ChatOpenAI

from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.utilities import GoogleSearchAPIWrapper

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = "Your Open AI API Key"
os.environ["GOOGLE_CSE_ID"] = "Your Google CSE ID"
os.environ["GOOGLE_API_KEY"] = "Your Google API Key"
os.environ["GPLACES_API_KEY"] = "Your Google Places API Key"


'''
template = """
As a local travel tour professional, you will answer your questions to the best of your ability. You have access to the following tools:

{tools}

Use the following format:

Question: The question you have to answer
Thought: Your thought process in approaching the question
Action: Choose one of the available tools in [{tool_names}] for your action
Action Input: Provide the input required for the chosen tool
Observation: Describe the result obtained from the action
...(Repeat several times of the Thought/Action/Action Input/Observation as needed)
Thought: Now I have the final answer!
Final Answer: Provide your final answer from the perspective of an experienced local travel tour professional

Let's get started!
Question: {input}
{agent_scratchpad}"""
'''
template = """Answer the following questions as best you can, but speaking as passionate travel expert. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have the final answer, and I will provide the address, phone number, and website if they exist.
Final Answer: The conclusive response to the original input question, including the data obtained from the used tool.

Begin! Remember to answer in brazilian portuguese as a passionate and informative travel expert when giving your final answer. 

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        print(thoughts)
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*:\s*(.*?)\s*Action\s*Input\s*:\s*(.*?)\s*(?:Observation:|$)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"{llm_output}")
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def search_general(input_text):
    search = GoogleSearchAPIWrapper(k=5).run(f"{input_text}")
    return search

def search_places(input_text):
    search = GooglePlacesTool().run(f"{input_text}")
    return search_general(search)

@cl.langchain_factory
def agent():
    tools = [
        Tool(
            name="Google Search",
            description="useful for when you need to answer general travel questions",
            func=search_general,
        ),
        Tool(
            name = "Search Trip Google",
            func=search_general,
            description="useful for when you need to answer trip plan questions"
        ),
        Tool(
            name = "Search places",
            func=search_places,
            description="useful when you need to answer informations about variety of places including hotels, restaurants, landmarks, businesses, geographical locations"
        ),
    ]
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()
    llm = ChatOpenAI(temperature=0.7)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]


    agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    verbose=True)
    return agent_executor

