from dotenv import load_dotenv, dotenv_values
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
load_dotenv()



restaurant_template = """
    I want you to act as a naming consultant for new restaurants.
    Return a list of restaurant names. Each name should be short, catchy and easy to remember. It should relate to the type of a restaurant you are naming. 
    What are some good names for a restaurant that is {restaurant_description}?
"""

llm = OpenAI(model_name = 'text-davinci-003',
             temperature = 0,
             max_tokens = 256)

prompt_template = PromptTemplate(input_variables=['restaurant_description'], template = restaurant_template)

description_1 = "a Greek place that serves fresh lamb souvlakis and other Greek "
description_2 = "a burger place that is themed with baseball memorabilia"
description_3 = "a cafe that has live hard rock music and memorabilia"

prompt_template.format(restaurant_description = description_1)

chain = LLMChain(llm=llm, prompt = prompt_template)

# print(chain.run(description_2))


examples = [
    {'word': 'happy', 'antonym': 'sad'},
    {'word': 'tall', 'antonym': 'short'}
]

example_formatter_template = """
    Word: {word}
    Antonym: {antonym}\n
"""

example_prompt = PromptTemplate(
    input_variables=['word', 'antonym'],
    template = example_formatter_template
)

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix='Give the antonym of every input',
    suffix='Word: {input}\nAntonym: ',
    input_variables=['input'],
    example_separator='\n\n'
)


chain = LLMChain(llm=llm, prompt=few_shot_template)
print(chain.run('Big'))



# text = 'Why did the chicken cross the road?'
# print(llm(text))
