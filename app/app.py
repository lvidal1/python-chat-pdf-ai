import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from htmlTemplate import bot_template, user_template

def main():
  load_dotenv()
  st.set_page_config(page_title="Chat with multiple PDF xdx", page_icon=":smiley:")

  if "conversation" not in st.session_state:
    st.session_state.conversation = None

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

  st.header("Chat with multiple PDF :books:")
  user_question = st.text_input("Ask a question about your doc")

  if user_question:
    handle_userinput(user_question)

  #st.write(bot_template.replace("{{message}}", "Hello"), unsafe_allow_html=True)
  #st.write(user_template.replace("{{message}}", "Hi"), unsafe_allow_html=True)

  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your documents", type=["pdf"], accept_multiple_files=True)

    if st.button("Ask"):
      with st.spinner("Processing"):
        # Get the pdf text
        raw_text = get_pdf_text(pdf_docs)

        # Get the text chunks
        text_chunks = get_text_chunks(raw_text)

        # Create vectors for each chunk
        vectorstore = get_vectorstore(text_chunks)

        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)


def get_pdf_text(pdf_docs):
  text = ""

  for pdf in pdf_docs:
    reader = PdfReader(pdf)
    for page in reader.pages:
      text += page.extract_text()

  return text

def get_text_chunks(raw_text):
  text_splitter = RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len
  )
  return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings()
  #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

def get_conversation_chain(vectorstore):
  llm = ChatOpenAI()
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  converation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
  )
  return converation_chain

def handle_userinput(user_question):
  if st.session_state.conversation is not None or st.session_state.chat_history is not None:
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']


    for i, msn in enumerate(st.session_state.chat_history):
      if i % 2 == 0:
        st.write(user_template.replace("{{message}}", msn.content), unsafe_allow_html=True)
      else:
        st.write(bot_template.replace("{{message}}", msn.content), unsafe_allow_html=True)

  else:
    response = "Impossible to retrieve the answer"
    st.write(response)


if __name__ == "__main__":
    main()