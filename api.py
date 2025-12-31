from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import base64
import tempfile
import requests
from gtts import gTTS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

# Cargar claves
load_dotenv()

app = FastAPI(title="Heimdall API V-Final", description="Backend con Reset")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEMORIA RAM ---
GLOBAL_RETRIEVER = None 

# --- AUXILIARES ---
def vision_hf_logic(imagen_bytes, prompt_usuario):
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token: return "Error: Falta Token HF"
    headers = {"Authorization": f"Bearer {token}"}
    modelos = [
        "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
        "https://api-inference.huggingface.co/models/microsoft/git-base"
    ]
    descripcion = None
    for url in modelos:
        try:
            res = requests.post(url, headers=headers, data=imagen_bytes)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and "generated_text" in data[0]:
                    descripcion = data[0]["generated_text"]
                    break
        except: continue
    if not descripcion: return "No pude ver la imagen."
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = f"CONTEXTO: Imagen.\nIA VE: '{descripcion}'\nPREGUNTA: '{prompt_usuario}'\nRESPONDE: Español."
    return llm.invoke(prompt).content

def generar_audio_base64(texto):
    try:
        if not texto: return None
        tts = gTTS(text=texto[:400], lang='es')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        return base64.b64encode(mp3_fp.getvalue()).decode('utf-8')
    except: return None

# --- ENDPOINTS ---

@app.post("/reset")
def reset_memory():
    """Borra la memoria del PDF y reinicia el estado"""
    global GLOBAL_RETRIEVER
    GLOBAL_RETRIEVER = None
    print("🧹 MEMORIA BORRADA: PDF eliminado del servidor.")
    return {"status": "ok", "message": "Memoria reiniciada."}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global GLOBAL_RETRIEVER
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        GLOBAL_RETRIEVER = vectorstore.as_retriever()
        
        os.remove(tmp_path)
        print("✅ PDF Cargado")
        return {"status": "ok", "message": "PDF procesado."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat_endpoint(
    prompt: str = Form(...), 
    image: UploadFile = File(None),
    personality: str = Form("Eres un asistente útil.")
):
    global GLOBAL_RETRIEVER
    respuesta = ""
    
    if image:
        print("📸 Modo Visión")
        contents = await image.read()
        respuesta = vision_hf_logic(contents, prompt)
    else:
        print("🧠 Modo Agente")
        try:
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
            tools = [TavilySearchResults(max_results=1)]
            pdf_notice = "No hay PDF cargado."
            
            if GLOBAL_RETRIEVER:
                tools.append(create_retriever_tool(GLOBAL_RETRIEVER, "search_pdf", "Busca información en el PDF."))
                pdf_notice = "HAY UN PDF CARGADO."

            plantilla_react = """
            Eres Heimdall. Herramientas: {tools}.
            REGLAS:
            1. Saludos/Identidad -> Responde DIRECTO sin herramientas.
            2. PDF/Internet -> Usa {tool_names}.
            3. ANTI-BUCLE: Si usas search_pdf y no está la info, NO BUSQUES DE NUEVO.
            4. JAMÁS dejes 'Action' vacío.

            FORMATO:
            Question: pregunta
            Thought: ¿Necesito herramienta?
            Action: herramienta (o vacío si respondes directo)
            Action Input: búsqueda
            Observation: resultado
            ...
            Final Answer: respuesta en ESPAÑOL.

            Personalidad: {personality}
            Contexto PDF: {pdf_notice}
            Question: {input}
            Thought:{agent_scratchpad}
            """
            prompt_template = PromptTemplate.from_template(plantilla_react)
            agent = create_react_agent(llm, tools, prompt_template)
            executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, max_iterations=4)
            res = executor.invoke({"input": prompt, "personality": personality, "pdf_notice": pdf_notice})
            respuesta = res["output"]
        except Exception as e:
            respuesta = "Error técnico: " + str(e)

    audio = generar_audio_base64(respuesta)
    return {"response": respuesta, "audio": audio}