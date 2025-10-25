#import Langchain Dependencies
from langchain.document_loaders import PyPDFLoader 
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# python -m streamlit run app.py

#streamlit for UI dev
import streamlit as st

#bring in watsonx dependencies
from langchain_ibm import ChatWatsonx

#create llm using longchain
llm = ChatWatsonx (
    model_id = 'ibm/granite-3-8b-instruct',
    project_id = '0763756d-c430-49e5-8705-eda1f8a616f7',
    url = 'https://au-syd.ml.cloud.ibm.com', 
    apikey = 'G8whaMJnXB1_Vcwff8ShElcWuL4gfxVV-dthPhTZxAVt',
    params = {
        "temperature": 0.7,
        "max_new_tokens": 200,
        "decoding_method": "sample"
    }
)

#load pdf of your choosing 
@st.cache_resource
def load_pdfs():
    pdf_files = [
        "Eco_Lifestyle_Guide.pdf",
        "Sustainable_Habits.pdf",
    ]
    
    loaders = [PyPDFLoader(pdf) for pdf in pdf_files]
    
    # Create vector database for all PDFs
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    ).from_loaders(loaders)
    #return vector database
    return index

#load the pdf into the index
index = load_pdfs()

#Create a retrieval based QA chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
)


#Setup the app title 
st.title("Eco Lifestyle Agent Guide")

#setup a session state message to hold chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

#Build a prompt imput template to display the elements 
prompt = st.text_input("Ask your Eco Lifestyle Agent a question:")  

#if user hits enter
if prompt:
    #display the prompt
    st.chat_message('user').markdown(prompt)
    #append the user prompt to the session state messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    #send the prompt to the llm model
    response = chain.run(prompt)
    #show the llm response
    st.chat_message('assistant').markdown(response)
    #append the llm response to the session state messages
    st.session_state.messages.append({"role": "assistant", "content": response})





