import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import warnings
import tempfile
import nest_asyncio

# Apply the patch to allow nested event loops
nest_asyncio.apply()

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")
# Load environment variables from a .env file
load_dotenv()

# Set up the Google API key from environment variables
# Make sure you have GOOGLE_API_KEY in your .env file
google_api_key = st.secrets["GOOGLE_API_KEY"]

os.environ["GOOGLE_API_KEY"] = google_api_key

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("Chat with your PDF using Gemini 1.5 Pro, LangChain, and Streamlit")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.success(f"PDF file '{uploaded_file.name}' uploaded successfully!")

    # --- Document Processing ---
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()

    # Split the document text into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    # --- Embeddings and Vector Store ---
    # Create embeddings using Google's model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create a FAISS vector store from the document chunks and embeddings
    vector_store = FAISS.from_documents(docs, embeddings)

    # --- Conversational Chain Setup ---
    # Set up memory to store chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize the Gemini 1.5 Pro model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    # --- Chat Interface ---
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Get user input
    query = st.text_input("Ask a question about the content of the PDF:")

    if query:
        # Get the response from the chain
        response = chain({"question": query, "chat_history": st.session_state.chat_history})
        
        # Add user query and AI response to the chat history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("AI", response['answer']))

    # Display the chat history
    st.write("---")
    st.write("### Conversation History")
    for speaker, text in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"<p style='text-align: right; color: #3498db;'><b>You:</b> {text}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: left; color: #2ecc71;'><b>AI:</b> {text}</p>", unsafe_allow_html=True)

    # Clean up the temporary file
    os.remove(temp_file_path)
