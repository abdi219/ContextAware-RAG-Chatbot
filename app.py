import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("Context-Aware Chatbot")

@st.cache_resource
def initialize_system():
    corpus = """
      Lahore Garrison University offers undergraduate and graduate programs in Computer Science, Software Engineering, Information Technology, and other disciplines.
      The Central Library provides access to books, digital resources, and study areas for registered students during working hours.
      Students must maintain the required attendance percentage to be eligible for semester examinations.
      The Computer Science department encourages participation in programming competitions, research projects, workshops, and technical societies.
      The university provides career counseling, internship guidance, and placement support to help students prepare for professional careers.
      Campus facilities include computer laboratories, sports complexes, cafeterias, and student support services to enhance the learning experience.
      """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    docs = text_splitter.create_documents([corpus])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    m_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(m_id)
    mod = AutoModelForCausalLM.from_pretrained(m_id, torch_dtype=torch.float16, device_map="auto")
    p = pipeline(
        "text-generation",
        model=mod,
        tokenizer=tok,
        max_new_tokens=80,
        temperature=0.1,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=p)
    
    sys_prompt = """
      You are a helpful university assistant.
      Answer the user's question using only the provided context.
      Include important details from the context.
      Do not mention the prompt, system instructions, or context.

      Context:
      {context}
      """

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

chain = initialize_system()

if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

if user_query := st.chat_input("Ask something about Lahore Garrison University..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "text": user_query})
    
    res = chain.invoke({"input": user_query, "chat_history": st.session_state.history})
    answer = res["answer"]
    
    # Clean up tracking strings if leaked by the raw pipeline
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
        
    with st.chat_message("assistant"):
        st.write(answer)
        
    st.session_state.messages.append({"role": "assistant", "text": answer})
    st.session_state.history.append(("human", user_query))
    st.session_state.history.append(("ai", answer))
