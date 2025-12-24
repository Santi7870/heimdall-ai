import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Heimdall AI", page_icon="👁️", layout="wide")

# Cargar llaves
load_dotenv()

# ==========================================
# 🎨 BARRA LATERAL
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=50)
    st.title("⚙️ Configuración")
    st.markdown("---")
    
    st.subheader("🎭 Personalidad")
    persona_default = "Eres un asistente útil y amable llamado Heimdall."
    user_persona = st.text_area(
        "Define su comportamiento:", 
        value=persona_default,
        height=100
    )
    
    if st.button("🔄 Aplicar Personalidad", use_container_width=True):
        if "agent_executor" in st.session_state:
            del st.session_state.agent_executor
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Borrar Chat", use_container_width=True):
        st.session_state.messages = []
        if "agent_executor" in st.session_state:
            st.session_state.agent_executor.memory.clear()
        st.rerun()

# ==========================================
# 🧠 LÓGICA PRINCIPAL
# ==========================================

st.title("👁️ Heimdall AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mensaje inicial
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "¡Hola! Soy Heimdall. ¿En qué puedo ayudarte hoy?"
    })

# --- CONFIGURACIÓN DEL CEREBRO ---
if "agent_executor" not in st.session_state:
    
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("❌ Faltan las claves en el archivo .env")
        st.stop()
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    tools = [TavilySearchResults(max_results=1)]
    
    # === PROMPT ANTI-BUCLES ===
    # Fíjate en la sección "RESPONSE FORMAT"
    template = f'''{user_persona}
    
    TOOLS:
    ------
    You have access to the following tools:
    {{tools}}

    IMPORTANT RULES:
    1. You have access to the conversation history.
    2. IF THE USER IS JUST CHATTING, ASKING FOR ADVICE, OR GREETING (e.g., "Hello", "How do I dissect an animal?", "I feel bad"), DO NOT USE THE SEARCH TOOL unless strictly necessary for facts.
    
    RESPONSE FORMAT (CHOOSE ONE):
    
    OPTION 1: IF YOU NEED TO SEARCH GOOGLE:
    Question: the input question
    Thought: I need to search for facts...
    Action: tavily_search_results_json
    Action Input: "search query"
    Observation: ...
    
    OPTION 2: IF NO SEARCH IS NEEDED (Chat, logic, refusals, greetings):
    Question: the input question
    Thought: I can answer this without tools...
    Final Answer: [Write your full answer here directly]

    WARNING: NEVER write "Action: None" or "Action: N/A". If no tool is needed, WRITE "Final Answer:" IMMEDIATELY after the Thought.

    Begin!

    Previous conversation:
    {{chat_history}}

    Question: {{input}}
    Thought:{{agent_scratchpad}}
    '''
    
    prompt = PromptTemplate.from_template(template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = create_react_agent(llm, tools, prompt)
    
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
    )

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input usuario
if prompt_user := st.chat_input("Escribe aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt_user})
    with st.chat_message("user"):
        st.markdown(prompt_user)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("🧠 Pensando..."):
            try:
                # El parámetro handle_parsing_errors=True ayuda a recuperar si falla el formato
                response = st.session_state.agent_executor.invoke({"input": prompt_user})
                output = response["output"]
                placeholder.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                # Si falla todo, mostramos el error limpio
                placeholder.error(f"❌ Error: {e}")