import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

from dotenv import load_dotenv
load_dotenv()

# Setup Streamlit App
st.set_page_config(page_title = 'Text to Math Problem Solver and Data Search Assistance', page_icon = '🧮')
st.title('Text to Math Promblem Solver using Google Gemma2')

groq_api_key = st.sidebar.text_input(label = 'Groq API Key', type = 'password')

if not groq_api_key:
    st.info('Please enter your Groq API Key to continue...')
    st.stop()
    
llm = ChatGroq(model = 'gemma2-9b-it', groq_api_key = groq_api_key)


# Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = 'Wikipedia',
    func = wikipedia_wrapper.run,
    description = 'A tool for searching the Internet to find the various information on the topic mention'
)

# Initialize the Math tool
math_chain = LLMMathChain.from_llm(llm = llm)
calculator = Tool(
    name = 'Calculator',
    func = math_chain.run,
    description = 'A tool for answer math related questions. Only input mathematical expression needs to be provided'
)

prompt = """
You are a agent tasked for solving users mathematical questions. 
Logically arrive at the solution and provide a detailed explanation and display it pointwise,
for the question below.
Question : {question}
Asnwer:
"""

prompt_template = PromptTemplate(
    input_variables = ['question'],
    template = prompt
)

# Combine all the tools into the chain
chain = LLMChain(llm = llm, prompt = prompt_template)


# Add Reasoning too
reasoning_tool = Tool(
    name = 'Reasoning',
    func = chain.run,
    description = 'A tool for answering logic-based and reasoning questions.'
)

# Initialize the agents
assistant_agent = initialize_agent(
    tools = [wikipedia_tool, calculator, reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    handle_parsing_errors = True
)

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            'role' : 'assistance',
            'content' : 'Hi, I am a math chatbot who can answer all your maths problem'
        }
    ]
    
    
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
    

question = st.text_area('Enter you question:', "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

# Lets start the interaction
if st.button('Find my answer'):
    if question:
        with st.spinner('Generating response...'):
            st.session_state.messages.append({'role' : 'user', 'content' : question})
            st.chat_message('user').write(question)
            
            st_cb = StreamlitCallbackHandler(st.container())
            response = assistant_agent.run(st.session_state.messages, callbacks = [st_cb])
            
            st.session_state.messages.append({'role' : 'assistance', 'content' : response})
            st.write(response)
    else:
        st.warning('Please enter the question')