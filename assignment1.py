# import packages
import streamlit as st
st.set_page_config(page_title="💕RAG Document Chat", page_icon="💕", layout="wide")


import streamlit as st
import os
import tempfile
import shutil
from typing import List

try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()



# API key (i run my individual API Key in the devcontainer)
api_key = os.environ.get("API_KEY", "")
if not api_key:
    st.error("API_KEY not found! Please set it in terminal: export API_KEY='your_key'")
    st.stop()


# connect to cornell api
try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.ai.it.cornell.edu",
    )
    llm = ChatOpenAI(
        model="openai.gpt-4o",
        temperature=0.2,
        api_key=api_key,
        base_url="https://api.ai.it.cornell.edu"
    )
    
except Exception as e:
    st.error(f"Client initialization error: {e}")
    st.stop()

# set page 
st.set_page_config(page_title="RAG Document Chat", page_icon="💕", layout="wide")

# save chat records, vector libraries, loaded file names, and temporary directory paths across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "uploaded_files_names" not in st.session_state:
    st.session_state.uploaded_files_names = []
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None



# function: record which file is being loaded


def load_document(file_path: str, file_type: str) -> List[Document]:
    """Load document with multiple encoding fallbacks"""
    st.write(f"🔍 Loading {file_type} file: {os.path.basename(file_path)}")
    
    # txt file
    try:
        if file_type == "txt":
    
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    st.write(f"   Trying encoding: {encoding}")
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    st.success(f"   Success with {encoding}")
                    return [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.warning(f"   Error with {encoding}: {e}")
                    continue


            
            
            st.write("   Using fallback: utf-8 with error handling")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]
           # pdf file 
        elif file_type == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            st.success(f"   PDF loaded: {len(docs)} pages")
            return docs
        else:
            st.error(f"   Unsupported file type: {file_type}")
            return []
        # when error     
    except Exception as e:
        st.error(f"   Error loading file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return []

#handle uploaded documents. if there are no files, remind and return
def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int):
    """Process uploaded documents"""
    st.write("---")
    st.write("###  Processing Documents")
    
    if not uploaded_files:
        st.warning("No files uploaded")
        return
    
    try:
        # create temp directory
        if st.session_state.temp_dir is None or not os.path.exists(st.session_state.temp_dir):
            st.session_state.temp_dir = tempfile.mkdtemp()
            st.write(f" Temp directory: {st.session_state.temp_dir}")
        
        temp_dir = st.session_state.temp_dir
        all_documents = []
        file_names = []
        
        # process each file
        st.write(f" Processing {len(uploaded_files)} file(s)...")
        
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            st.write(f"\n**File {idx}/{len(uploaded_files)}: {uploaded_file.name}**")
            
            try:
                # sanitize filename
                file_extension = uploaded_file.name.split('.')[-1].lower()
                safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in ('_', '-', '.'))
                if not safe_filename:
                    safe_filename = f"file_{idx}.{file_extension}"
                
                temp_file_path = os.path.join(temp_dir, safe_filename)
                st.write(f" Saving as: {safe_filename}")
                
                # save file
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write(f" File saved ({os.path.getsize(temp_file_path)} bytes)")
                
                # load document
                docs = load_document(temp_file_path, file_extension)
                
                if docs:
                   
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    all_documents.extend(docs)
                    file_names.append(uploaded_file.name)
                    st.success(f" Loaded {len(docs)} document(s)")
                else:
                    st.warning(f" No content loaded from {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f" Error with {uploaded_file.name}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                continue
        
        if not all_documents:
            st.error(" No documents were successfully loaded")
            return
        
        st.write("---")
        st.write(f"###  Chunking {len(all_documents)} document(s)")
        
        # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_documents)
        st.success(f" Created {len(chunks)} chunks")
        
        # embeddings
        st.write("###  Creating embeddings...")
        embeddings = OpenAIEmbeddings(
            model="openai.text-embedding-3-large",
            api_key=api_key,
            base_url="https://api.ai.it.cornell.edu"
        )
        
        # vector store
        st.write("###  Building vector database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="document_collection"
        )
        
        # save to session state
        st.session_state.vectorstore = vectorstore
        st.session_state.uploaded_files_names = file_names
        
        st.write("---")
        st.success(f" Successfully processed {len(file_names)} document(s)!")
        
    except Exception as e:
        st.error(f" Processing error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def retrieve_and_generate(question: str, k: int = 5) -> dict:
    """Retrieve and generate answer"""
    if st.session_state.vectorstore is None:
        return {"answer": "Please upload and process documents first before you ask the question 🙋🏼‍♀️", "sources": []}
    
    try:
        # retrieve
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        retrieved_docs = retriever.invoke(question)
        
        # format context
        context = "\n\n".join([
            f"[From: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        # prompt
        template = """
You are an assistant designed to answer questions using the provided context.
Base your response strictly on the information retrieved.
If the context does not contain the answer, respond with: "I’m not sure based on the given information 🥹."
Provide clear and concise answers.

Context:
{context}



Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        messages = prompt.invoke({"question": question, "context": context})
        
        # Generate
        response = llm.invoke(messages)
        
        # Extract sources
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
        
        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_chunks": len(retrieved_docs)
        }
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return {"answer": f"An error occurred: {str(e)}", "sources": []}



# UI design section


st.title(" 💕RAG Document Chat💕")
st.markdown("This version shows detailed processing steps")

# sidebar
with st.sidebar:
    st.header(" Settings")
    
    chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    k_documents = st.slider("Retrieved Chunks", 1, 20, 5, 1)
    
    chunk_strategy = st.selectbox(
    "Chunking Strategy",
    ["Recursive (default)", "By Paragraph (\\n\\n)", "PDF: one page = one chunk"],
    index=0
)



    st.divider()
    
    st.header(" Upload Documents")
    
    if st.session_state.uploaded_files_names:
        st.subheader(" Loaded Documents")
        for filename in st.session_state.uploaded_files_names:
            st.text(f"✓ {filename}")
    
    if st.button(" Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button(" Reset All"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.uploaded_files_names = []
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        st.session_state.temp_dir = None
        st.rerun()

# main interaction section
if not st.session_state.vectorstore:
    st.info(" Please upload the document first before you ask the question 🙋🏼‍♀️")
else:
    st.success(f" Ready! {len(st.session_state.uploaded_files_names)} document(s) loaded")
uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Upload .txt or .pdf files"
    )
if st.button("Process Documents", disabled=not uploaded_files):
        process_documents(uploaded_files, chunk_size, chunk_overlap)

# chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(" Sources"):
                st.write(f"Retrieved {msg.get('retrieved_chunks', 0)} chunks from:")
                for source in msg["sources"]:
                    st.write(f"• {source}")

# chat input
if question := st.chat_input("Ask a question...", disabled=not st.session_state.vectorstore):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = retrieve_and_generate(question, k=k_documents)
            st.write(result["answer"])
            
            if result["sources"]:
                with st.expander(" Sources"):
                    st.write(f"Retrieved {result.get('retrieved_chunks', 0)} chunks from:")
                    for source in result["sources"]:
                        st.write(f"• {source}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "retrieved_chunks": result.get("retrieved_chunks", 0)
    })
