from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
import os
import openai
from dotenv import load_dotenv

from llama_index import GPTSimpleVectorIndex, download_loader
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
import streamlit as st


load_dotenv()
openai.key = os.getenv("OPENAI_API_KEY")
@st.cache_resource
def generate_roadmap(urls):
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=urls)
    index = GPTSimpleVectorIndex.from_documents(documents)
    print(index)
    tools = [
        Tool(
            name="Website Index",
            func=lambda q: index.query(q),
            description=f"Useful when you want answer questions about the text on websites.",
        ),
    ]
    llm = OpenAI(temperature=0)
    output = index.query("What is the Company about?")
    # clients_output = index.query("Provide a detailed overview of the company's clients it operates with.")
    
    clients_output = index.query("Provide a summary of the mostly frequently asked questions about the company")# output += detailed_summary

    # memory = ConversationBufferMemory(memory_key="chat_history")
    # agent_chain = initialize_agent(
    #     tools, llm, agent="zero-shot-react-description", memory=memory
    # )

    # output = agent_chain.run(input="What is the Company about? What kind of services does it deal with?")
    
    prompt = f'''
    Imagine you have a PhD in Machine learning, and a MBA as well. 
    You understand product market fit really well, and also see the pain points when provided a company description and their workings.
    You are also very well versed in generative ai technologies and have sucessfully founded 4 AI startups in the last few years. Given your breadth of experience, one you are given a company url, you need to first identify how generative ai can be useful for the company and then suggest a product roadmap incorporating your plans on integration. Be detailed yet concise, and don’t give generic advice.
    Give highly professional company-specific advice on the roadmap.
    Be detailed yet concise, and don’t give generic advice. 
    Give highly professional company-specific advice on the roadmap.

    Company Website Description: 
    {output}
    {clients_output}

    Roadmap:
    '''
    summary = str(output) + '\n' + str(clients_output)

    system_text = "You are really talented product manager who understands different companies very well. You can tailor AI solutions on the basis of their company description and help formulate product roadmaps. "
    chat_query = [{"role":"system", "content": system_text}, {"role":"user", "content": prompt}]
    response = openai.ChatCompletion.create(
        messages=chat_query,
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=' ;'
        )    
    return summary, response["choices"][0]["message"]["content"]

# if main
if __name__ == '__main__':

    # user_input = input("Enter a url: ")

    user_input = st.text_input("Enter a url: ", key =999)

    if st.button('Submit'):
        with st.spinner():
            output, response = generate_roadmap([user_input])

        if response is not None:
            st.balloons()
            st.markdown('### Product Roadmap')
            st.write(response)
            with st.expander('See Website Description'):
                st.write(output)
    
    # print("WEBSITE_INFO \n",output,'\n','--- RESPONSE --- \n', response)