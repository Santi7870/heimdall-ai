import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Heimdall AI", page_icon="👁️", layout="wide")

# --- TRUCO PARA LA NUBE: Cargar Secretos al Entorno ---
# Esto hace que funcione tanto en tu PC (.env) como en la Nube (Secrets)
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

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
    
    # Verificación final de claves
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        st.error("❌ Error: No se encontraron las API KEYS. Revisa los 'Secrets' en Streamlit Cloud.")
        st.stop()
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    tools = [TavilySearchResults(max_results=1)]
    
    # === PROMPT CORREGIDO (CON TOOL_NAMES) ===
    template = f'''{user_persona}
    
    TOOLS:
    ------
    You have access to the following tools:
    {{tools}}
    
    IMPORTANT RULES:
    1. IF THE USER IS JUST CHATTING (e.g., "Hello", "How are you?"), DO NOT USE TOOLS.
    2. USE THE FORMAT BELOW EXACTLY.
    
    FORMAT INSTRUCTIONS:
    To use a tool, please use the following format:
    
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{{tool_names}}] 
    Action Input: the input to the action
    Observation: the result of the action
    
    If you do not need to use a tool, use this format:
    
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    
    Begin!

    Previous conversation:
    {{chat_history}}

    Question: {{input}}
    Thought:{{agent_scratchpad}}
    '''
    
    # Aquí es donde fallaba antes, ahora ya tiene tool_names
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
                response = st.session_state.agent_executor.invoke({"input": prompt_user})
                output = response["output"]
                placeholder.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                placeholder.error(f"❌ Error: {e}")