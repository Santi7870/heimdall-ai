import streamlit as st
import os
import tempfile
import base64
import requests
import time
import io
from gtts import gTTS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage

# Importaciones para PDF (RAG)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Heimdall AI", page_icon="👁️", layout="wide")

# --- ESTADO INICIAL ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- CARGA DE CLAVES ---
load_dotenv()
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    pass

# ==========================================
# 🧠 MEMORIA
# ==========================================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

# ==========================================
# 🛠️ FUNCIONES AUXILIARES
# ==========================================
def reiniciar_agente():
    if "agent_executor" in st.session_state:
        del st.session_state.agent_executor

def reset_nuclear():
    st.session_state.messages = []
    st.session_state.memory.clear()
    if "agent_executor" in st.session_state:
        del st.session_state.agent_executor
    st.session_state.uploader_key += 1
    st.rerun()

# --- NUEVA FUNCIÓN DE VOZ ---
def generar_audio(texto):
    """Convierte texto a audio MP3 en memoria usando gTTS."""
    if not texto: return None
    try:
        # Solo generamos audio si el texto no es muy largo para no trabar la app
        if len(texto) > 500:
            texto = texto[:500] + "..." # Recortamos para demo rápida
            
        tts = gTTS(text=texto, lang='es')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Error de audio: {e}")
        return None

# --- FUNCIÓN DE VISIÓN BLINDADA ---
def vision_hf(image_file, prompt_usuario):
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token: return "❌ Error: Falta Token HF."
    headers = {"Authorization": f"Bearer {token}"}
    imagen_bytes = image_file.read()
    
    modelos = [
        "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
        "https://api-inference.huggingface.co/models/microsoft/git-base",
        "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    ]
    descripcion_tecnica = None

    for i, url in enumerate(modelos):
        try:
            response = requests.post(url, headers=headers, data=imagen_bytes)
            if response.status_code == 503:
                time.sleep(2)
                response = requests.post(url, headers=headers, data=imagen_bytes)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and "generated_text" in data[0]:
                    descripcion_tecnica = data[0]["generated_text"]
                    break 
        except Exception: continue

    if not descripcion_tecnica:
        return "❌ Servidores de visión ocupados. Intenta en 1 min."

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = f"CONTEXTO: Imagen enviada.\nIA VE (Inglés): '{descripcion_tecnica}'\nPREGUNTA: '{prompt_usuario}'\nINSTRUCCIÓN: Responde en español."
    return llm.invoke(prompt).content

# ==========================================
# 🧠 RAG (PDF)
# ==========================================
@st.cache_resource
def procesar_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.remove(tmp_path)
    return vectorstore.as_retriever()

# ==========================================
# 🎨 UI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=50)
    st.title("⚙️ Configuración")
    st.markdown("---")
    
    st.subheader("🎭 Personalidad")
    user_persona = st.text_area("Comportamiento:", value="Eres un asistente útil y amable llamado Heimdall.", height=100, key=f"p_{st.session_state.uploader_key}")
    if st.button("🔄 Aplicar", use_container_width=True):
        reiniciar_agente()
        st.rerun()

    st.markdown("---")
    st.subheader("📄 Documentos")
    uploaded_pdf = st.file_uploader("PDF", type="pdf", on_change=reiniciar_agente, key=f"pdf_{st.session_state.uploader_key}")
    
    st.markdown("---")
    st.subheader("📸 Cámara/Imagen")
    uploaded_image = st.file_uploader("Imagen", type=["jpg","png"], key=f"img_{st.session_state.uploader_key}")
    
    st.markdown("---")
    if st.button("🗑️ RESET TOTAL", type="primary", use_container_width=True):
        reset_nuclear()

# ==========================================
# 🧠 MAIN
# ==========================================
st.title("👁️ Heimdall AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "¡Hola! Soy Heimdall. Te escucho."})

# AGENTE SETUP
if "agent_executor" not in st.session_state:
    if not os.environ.get("GROQ_API_KEY"): st.stop()
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    tools = [TavilySearchResults(max_results=1)]
    
    pdf_instr = ""
    if uploaded_pdf:
        with st.spinner("📚 Leyendo PDF..."):
            retriever = procesar_pdf(uploaded_pdf)
            tools.append(create_retriever_tool(retriever, "buscar_pdf", "Busca en el PDF."))
            st.toast("PDF Listo", icon="✅")
            pdf_instr = "URGENT: PDF UPLOADED. USE 'buscar_pdf' TOOL."

    # --- AQUI ESTABA EL ERROR: RESTAURADO EL PROMPT COMPLETO ---
    template = f'''{user_persona}
    
    SYSTEM NOTICE: {pdf_instr}
    
    TOOLS:
    ------
    You have access to the following tools:
    {{tools}}
    
    To use a tool, please use the following format:
    
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{{tool_names}}]
    Action Input: the input to the action
    Observation: the result of the action
    
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
    
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    
    Begin!

    Previous conversation:
    {{chat_history}}

    Question: {{input}}
    Thought:{{agent_scratchpad}}
    '''
    
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, memory=st.session_state.memory)

# CHAT DISPLAY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# INPUT HANDLING
if prompt_user := st.chat_input("Escribe aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt_user})
    with st.chat_message("user"): st.markdown(prompt_user)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        # LOGICA CEREBRO
        output = ""
        if uploaded_image:
            if uploaded_pdf: st.warning("⚠️ Ignorando PDF por Imagen.")
            with st.spinner("👁️ Viendo..."):
                uploaded_image.seek(0)
                output = vision_hf(uploaded_image, prompt_user)
        else:
            with st.spinner("🧠 Pensando..."):
                try:
                    res = st.session_state.agent_executor.invoke({"input": prompt_user})
                    output = res["output"]
                except Exception as e:
                    output = f"Error: {e}"
        
        # MOSTRAR Y GUARDAR
        placeholder.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
        
        # 🔊 GENERAR AUDIO AUTOMÁTICO
        if output and not output.startswith("Error") and not output.startswith("❌"):
            audio_file = generar_audio(output)
            if audio_file:
                st.audio(audio_file, format="audio/mp3", autoplay=True)