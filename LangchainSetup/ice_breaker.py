import os
from dotenv import load_dotenv, dotenv_values
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
import requests
import json
from agents.linkedin_lookup_agent import lookup

load_dotenv()

if __name__ == "__main__":
    print("Hello Langchain!")
    
    linkedin_profile_url = lookup(name='Olga Sandu')

    summary_template = """
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(   
        input_variables=["information"],
        template=summary_template
        )
    
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    chain = LLMChain(llm=llm, prompt = summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)
    
    print(chain.run(information=linkedin_data))
