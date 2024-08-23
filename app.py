import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit_pills import pills
from src.helpers import DocumentProcessor
import os
import tempfile
#from src.helpers import get_similar_docs, format_as_context
import psycopg2

# app config
st.set_page_config(page_title="My-Rag", page_icon="🤖")

OPENAI_API_TYPE=os.getenv("OPENAI_API_TYPE")
OPENAI_API_VERSION=os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_EMB=os.getenv("AZURE_DEPLOYMENT_EMB")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

#connection_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}'
#db_connection = psycopg2.connect(connection_string)
llm=ChatOllama(model="llama3", temperature=0)
processor=DocumentProcessor()

def get_response(user_query, chat_history, llm):
    """Generates a response to a user query using the Ollama Llama3 model, considering the chat history and relevant context.
    Description:
    This function constructs a prompt for the AI assistant to generate a response based on the current user query and the chat history.
    It uses a pre-defined template to incorporate the chat history and user query, and retrieves relevant context using a similarity 
    search function (`get_similar_docs`). The retrieved context is formatted and included in the prompt. The Ollama Llama3 model is 
    then invoked to generate the response, which is streamed back to the user.
    
    Steps:
    1. Define the prompt template incorporating the chat history, user query, and additional context.
    2. Retrieve relevant documents or context related to the user query using `get_similar_docs`.
    3. Format the retrieved context for inclusion in the prompt.
    4. Initialize the Ollama Llama3 model with the appropriate settings.
    5. Create a processing chain using the prompt template, the language model, and the output parser.
    6. Generate and return the AI model's response by streaming the output based on the constructed chain.

    :param user_query (str): The current query or question posed by the user.
    :param chat_history (list): A list of previous chat messages. Each message is either an AIMessage or a HumanMessage.
    :return str: The generated response from the AI model, streamed incrementally.

    Example Usage:
    response = get_response("What is the capital of France?", chat_history)
    """
    template = """
        You are a helpful assistant. Answer the following questions considering the history of the conversation:

        Chat history: {chat_history}

        Answer the given question: {user_question}
        to answer the question only refer to the context i am now providing to you. If you dont find the information you need in the context, please tell me you cant answer. This is the context: {context}
        """
    prompt = ChatPromptTemplate.from_template(template)
    # context = get_similar_docs(user_query, conn=db_connection, threshold=0.75, n=1)
    # context = format_as_context(context)
    docs=db.similarity_search(user_query, k=1)
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
    chain = prompt | llm | StrOutputParser()
   
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "context": context
        })


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Upload your PDF to start asking a question to it"),
    ]

# Initialize file upload status in session state
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "db" not in st.session_state:
    st.session_state.db = None

# Show the file uploader if no file has been uploaded yet
if not st.session_state.file_uploaded:
    st.title("🏠 Upload your PDF to proceed!")
    uploaded_file = st.file_uploader("Upload your PDF", accept_multiple_files=False, label_visibility="hidden")
    if uploaded_file:
        st.session_state.file_uploaded = True
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        #print(temp_file_path)
        docs=processor.load_split_pdf(temp_file_path)
        print(docs)
        processor.create_and_save_db(docs)
        st.session_state.db = processor.load_db()
        st.session_state.uploaded_file = uploaded_file
        st.rerun()  # Rerun the app to hide the uploader

# If a file has been uploaded, proceed with the rest of the app
if st.session_state.file_uploaded:

    # Display the uploaded file name (optional)
    st.write(f"Uploaded file: {st.session_state.uploaded_file.name}")
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)

    # User input
    user_query = st.chat_input("How can I help?")

    # Show pills only if no user query is provided and no pill is selected
    if user_query is None and "selected" not in st.session_state:
        selected = pills("Document uploaded successfully, you can start with some example questions", ["Summarize the document", "What is the most important KPI", "what should i take away"], ["📄", "📎", "🎒"], index=None, clearable=True)
        if selected:
            st.session_state.selected = selected
            st.rerun()
    else:
        selected = st.session_state.get("selected", None)

    # Process user query or selected pill
    if (user_query is not None and user_query != "") or selected is not None:
        if selected is not None:
            user_query = selected

        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            db = st.session_state.db
            response = st.write_stream(get_response(user_query, st.session_state.chat_history, llm))

            st.session_state.chat_history.append(AIMessage(content=response))

        if "selected" in st.session_state:
            del st.session_state.selected