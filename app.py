import streamlit as st
import os
import torch
import pdfplumber
import time
from typing import TypedDict
from PIL import Image

# Pinecone & LangChain
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Transformers
from transformers import (
    AutoProcessor, 
    Qwen2VLForConditionalGeneration, 
    BitsAndBytesConfig
)

# 1. APP CONFIGURATION & ENHANCED UI STYLING
st.set_page_config(
    page_title="MatScribe: AI Materials Researcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme CSS 
st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        background-color: #0b0e11;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Card-like containers for chat */
    .stChatMessage {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #c9d1d9 !important;
        font-weight: 600 !important;
    }
    
    /* Custom button styling */
    .stButton>button {
        border-radius: 4px !important;
        border: 1px solid #30363d !important;
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #30363d !important;
        border-color: #8b949e !important;
    }
    
    /* Status indicator refinement */
    .stStatus {
        border: 1px solid #30363d !important;
        background-color: #0d1117 !important;
    }

    /* Input area styling */
    .stChatInput {
        border-top: 1px solid #30363d !important;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# 2. SIDEBAR CONFIGURATION
with st.sidebar:
    st.title("MatScribe")
    st.caption("Advanced Material Intelligence")
    st.markdown("---")
    
    env_key = os.getenv("PINECONE_API_KEY", "")
    
    if env_key:
        st.info("System: Pinecone authenticated via environment.")
        api_key = env_key
    else:
        api_key = st.text_input("Pinecone API Key", type="password")

    st.markdown("---")
    st.subheader("Configuration")
    use_vision = st.toggle("Vision Analysis", value=False, 
                          help="Analyzes charts and microstructures. Requires more compute.")
    
    if use_vision:
        st.warning("Mode: Deep Vision Scan Active")
    else:
        st.info("Mode: High-Speed Text Only")

    st.markdown("---")
    if st.button("Clear Session"):
        st.session_state.chat_history = []
        st.session_state.vectorstore_ready = False
        st.session_state.processing_complete = False
        st.rerun()

    st.markdown("### Documentation")
    st.caption("MatScribe utilizes agentic workflows to extract and cross-verify materials data from scientific publications.")

if not api_key:
    st.warning("Action Required: Please provide a Pinecone API Key in the sidebar.")
    st.stop()

os.environ["PINECONE_API_KEY"] = api_key
INDEX_NAME = "materials-project"

# 3. MODEL LOADING (CACHED)
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return processor, model, embed_model, device

if "model_loaded" not in st.session_state:
    with st.spinner("Initializing System Engines..."):
        processor, model, embed_model, device = load_models()
        st.session_state.model_loaded = True
else:
    processor, model, embed_model, device = load_models()

# 4. CORE LOGIC FUNCTIONS

def query_model(prompt, image=None, max_tokens=200):
    if image:
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    else:
        content = [{"type": "text", "text": prompt}]

    messages = [{"role": "user", "content": content}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_prompt], 
        images=[image] if image else None, 
        padding=True, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.split("assistant")[-1].strip() if "assistant" in response else response

def ingest_pdf(uploaded_file, enable_vision):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    pc = Pinecone(api_key=api_key)
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME, 
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)

    docs = []
    metadatas = []
    
    with pdfplumber.open("temp.pdf") as pdf:
        total_pages = len(pdf.pages)
        progress_bar = st.progress(0, text="Analyzing pages...")
        
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            progress_bar.progress(int((i / total_pages) * 100))
            
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_str = ""
            if tables:
                for t in tables:
                    for row in t:
                        clean_row = [str(c).replace('\n', ' ') if c else "" for c in row]
                        table_str += "| " + " | ".join(clean_row) + " |\n"
            
            img_desc = ""
            if enable_vision:
                for img in page.images:
                    if img['width'] < 150 or img['height'] < 150: continue
                    try:
                        cropped = page.crop((img['x0'], img['top'], img['x1'], img['bottom']))
                        img_obj = cropped.to_image(resolution=150).original
                        desc = query_model("Extract technical data from this figure.", image=img_obj)
                        img_desc += f"[Image Data]: {desc}\n"
                    except: pass
            
            full_content = f"PAGE {page_num}\nTEXT:\n{text}\nTABLES:\n{table_str}\nFIGURES:\n{img_desc}"
            docs.append(full_content)
            metadatas.append({"page": page_num, "source": uploaded_file.name})
            
        progress_bar.empty()
            
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embed_model)
    vectorstore.add_texts(docs, metadatas=metadatas)
    return vectorstore

# 5. MAIN UI LAYOUT
st.title("MatScribe")
st.markdown("### Agentic Materials Discovery and Verification")

#  A. FILE UPLOAD SECTION 
with st.container():
    uploaded_file = st.file_uploader("Input: Research Publication", type="pdf")

if uploaded_file and not st.session_state.vectorstore_ready:
    if st.button("Initialize Processing", type="primary"):
        with st.status("System Status: Processing Document", expanded=True) as status:
            status.write("Parsing document structure...")
            
            if use_vision:
                status.write("Vision Subsystem: Enabled")
            
            try:
                vectorstore = ingest_pdf(uploaded_file, use_vision)
                st.session_state.vectorstore_ready = True
                st.session_state.processing_complete = True
                status.update(label="System Status: Knowledge Base Ready", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label="System Status: Processing Error", state="error")
                st.error(f"Error Log: {str(e)}")

#  B. CHAT INTERFACE 
if st.session_state.vectorstore_ready:
    st.divider()
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embed_model)

    # Chat Display Container
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Fixed Bottom Input with Multi-Agent Logic
    if prompt := st.chat_input("Query material properties or experimental data...", key="chat_input_unique"):
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.status("Agent Workflow: Processing Query", expanded=True)
            
            # Node 1: Enhanced Retriever (k=7 to ensure broad context)
            status_container.write("Agent: Retrieving relevant context segments...")
            results = vectorstore.similarity_search(prompt, k=7)
            # Tagging context with source page to help Auditor attribution
            context = "\n\n".join([f"SOURCE PAGE {d.metadata['page']}: {d.page_content}" for d in results])
            
            # Node 2: Selective Extractor
            status_container.write("Agent: Performing data extraction...")
            extract_prompt = f"""
            Context: {context[:4000]}
            Question: {prompt}
            Task: Provide a technical answer based strictly on the context. 
            If the question asks for a general definition (e.g. HEA definition), look for text describing a class of materials.
            If it asks for specific values (e.g. Ms, Hc), look for specific alloy sample data.
            Crucial: Do not use specific sample properties to define a general material category.
            """
            extraction = query_model(extract_prompt)
            
            # Node 3: Attribution Auditor
            status_container.write("Agent: Verifying claim attribution...")
            audit_prompt = f"""
            Source Context: {context[:4000]}
            Extracted Claim: {extraction}
            Task: Is this claim correctly attributed? 
            Check: If the claim provides properties for a specific alloy (like FeCoNi) but the user asked for a general material definition (like HEA), answer FALSE.
            Answer ONLY 'TRUE' or 'FALSE'.
            """
            critique = query_model(audit_prompt, max_tokens=10)
            
            if "true" in critique.lower():
                status_container.update(label="Status: Verified Answer Found", state="complete", expanded=False)
                final_response = f"**{extraction}**\n\n*Source Verification: Confirmed*"
            else:
                status_container.update(label="Status: Attribution Warning", state="error", expanded=False)
                final_response = f"**{extraction}**\n\n*Source Verification: Inferred or Unverified. The system may be misattributing specific sample data to a general query.*"
            
            st.markdown(final_response)
            st.session_state.chat_history.append({"role": "assistant", "content": final_response})

elif not uploaded_file:
    st.info("System Ready. Please upload a PDF to begin ingestion.")