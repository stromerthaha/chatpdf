import streamlit as st
import pickle
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template  # Import templates

# Sidebar contents
st.sidebar.title('🤗💬 LLM Chat App')
st.sidebar.markdown('''
## About
This app is an LLM-powered chatbot built using:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://platform.openai.com/docs/models) LLM model
\n\n
Made by [Mohammed Thaha](https://github.com/stromerthaha/)
''')

def main():
    st.header("Chat with PDF 🤔💬")
    load_dotenv()
    st.markdown(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Upload a PDF file
    pdf_files = st.file_uploader("Upload your PDF", type='pdf', accept_multiple_files=True)

    if pdf_files:
        for pdf in pdf_files:
            # Save the PDF locally
            with open(f"temp_{pdf.name}", "wb") as temp_pdf:
                temp_pdf.write(pdf.read())

            # Read the PDF using PdfReader
            pdf_reader = PdfReader(f"temp_{pdf.name}")

            # Extract text from the PDF
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Embeddings
            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    vectorstore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(vectorstore, f)

            query = st.text_input("Ask questions about your PDF file:")

            if query:
                docs = vectorstore.similarity_search(query=query, k=3)

                # Check if conversation_chain is not initialized
                if st.session_state.conversation is None:
                    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever(),
                        memory=memory
                    )

                with get_openai_callback() as cb:
                    response = st.session_state.conversation({'question': query})
                    st.session_state.chat_history = response['chat_history']

                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            # Remove the temporary PDF file
            os.remove(f"temp_{pdf.name}")

if __name__ == '__main__':
    main()
