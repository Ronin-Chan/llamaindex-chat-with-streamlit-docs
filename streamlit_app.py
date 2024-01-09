import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from openai import OpenAI as OriginalOpenAI
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with CiiLOCK chatbot", page_icon="⚙️", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
client = OriginalOpenAI(
        # This is the default and can be omitted
        api_key=st.secrets.openai_key,
    )
st.title("Chat with CiiLOCK chatbot, powered by LlamaIndex")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about CiiLOCK!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs - hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts - do not hallucinate features. If you do not know the answer, respond with \"Sorry, I don't know.\""))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            if ("sorry" in response.response or "Sorry" in response.response or "don't know" in response.response or "Don't know" in response.response):
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="gpt-3.5-turbo-1106",
                )
                response.response = chat_completion.choices[0].message.content
                st.write("[Internal] " + response.response)
                message = {"role": "assistant", "content": "[Internal] " + response.response}
            else:
                st.write("[External] " + response.response)
                message = {"role": "assistant", "content": "[External] " + response.response}
            st.session_state.messages.append(message) # Add response to message history
