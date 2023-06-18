import os
from dotenv import load_dotenv, dotenv_values
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

if __name__ == "__main__":
    print("Hello Langchain!")
    
    summary_template = """
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables="information", template=summary_template)