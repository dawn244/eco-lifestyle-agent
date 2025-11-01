#import Langchain Dependencies
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
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
   model_id = st.secrets["MODEL_ID"],
    project_id = st.secrets['PROJECT_ID'],
    url = st.secrets['URL'], 
    apikey = st.secrets['API_KEY'],
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
    
    loader= [PyPDFLoader(pdf) for pdf in pdf_files]
    
    # Create vector database for all PDFs
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vectorstore
    db = FAISS.from_documents(texts, embeddings)
    index = db
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
st.set_page_config(page_title="Eco Lifestyle Agent", page_icon="🌿", layout="wide")

# Theme colors
#primaryColor="#a8d5ba"       # light green
#backgroundColor="#ffffff"    # white
#secondaryBackgroundColor="#d0f0f7"  # sea blue
#font="sans serif"


#App Header
st.markdown(
    """
    <h1 style='text-align: center; color: #2e7d32;'>🌿 Eco Lifestyle Agent Guide</h1>
    <p style='text-align: center; color: #0077b6;'>Ask about sustainable living tips, eco-friendly habits, and more!</p>
    """,
    unsafe_allow_html=True
)

#setup a session state message to hold chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display the chat history in blocks 
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style='text-align: right; background-color: #a8d5ba; color: #000000; padding: 10px; 
                    border-radius: 10px; margin: 5px; display: inline-block;'>{msg["content"]}</div>
        """, unsafe_allow_html=True)
    else:
        points = msg["content"].split("\n")
        for point in points:
            if point.strip() != "":
                st.markdown(f"""
                <div style='text-align: left; background-color: #d0f0f7; color: #000000; padding: 10px; 
                            border-radius: 10px; margin: 5px; display: inline-block;'>{point}</div>
                """, unsafe_allow_html=True)

#Build a prompt imput template to display the elements 
prompt = st.text_input("Ask for eco-friendly tips (e.g., How can I save energy at home? 🌎:")

if prompt:
    # Display prompt
    #st.chat_message('user').markdown(prompt)
    #append the user prompt to the session state messages
    st.session_state.messages.append({"role": "user", "content": prompt})
   
    # ♻️ Environmental system prompt
    eco_prompt = f"""
    You are an environmental lifestyle expert who helps people live sustainably.
    Provide eco-friendly advice based on the question below, using real and practical solutions.

    Be warm, encouraging, and informative. Avoid generic statements.

    Question: {prompt}

    Respond with:
    - Practical eco-tips or sustainable alternatives
    - Simple steps to implement
    - Optional ‘Did You Know?’ eco-fact
    """

    # Generate response using retrieval-based QA chain
    response = chain.run(eco_prompt)

    #llm response
    response = chain.run(prompt) 
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display user message immediately
    st.markdown(f"""
    <div style='text-align: right; background-color: #a8d5ba; color: #000000; padding: 10px; 
                border-radius: 10px; margin: 5px; display: inline-block;'>{prompt}</div>
    """, unsafe_allow_html=True)


    # Display assistant response immediately
    points = response.split("\n")
    for point in points:
        if point.strip() != "":
            st.markdown(f"""
            <div style='text-align: left; background-color: #d0f0f7; color: #000000; padding: 10px; 
                        border-radius: 10px; margin: 5px; display: inline-block;'>{point}</div>
            """, unsafe_allow_html=True)



