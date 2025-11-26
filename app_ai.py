import os
import json
import streamlit as st
from litellm import completion
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import pdfplumber
import uuid
import tempfile
from tavily import TavilyClient
from typing import List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


def _env_base(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip().rstrip("/")
    return default

# Cargar instrucciones
with open("agents_es.md", "r", encoding="utf-8") as f:
    agents_md = f.read()

st.set_page_config(page_title="Agente Plan - AI", layout="wide")

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL_ID", "ollama/Qwen3:8B")
DEFAULT_OLLAMA_BASE = _env_base("OLLAMA_API_BASE", "http://localhost:11434")
DEFAULT_GITHUB_MODEL = os.environ.get("GITHUB_MODEL_ID", "gpt-4o-mini")
DEFAULT_GITHUB_BASE = _env_base("GITHUB_MODELS_API_BASE", "https://api.githubcopilot.com/v1")
DEFAULT_CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL_ID", "cerebras/qwen-3-32b")
DEFAULT_CEREBRAS_BASE = _env_base("CEREBRAS_API_BASE", "https://api.cerebras.ai/v1")
PROVIDER_OPTIONS = ("Ollama (local)", "GitHub Models", "Cerebras")
MAX_TOOL_CALLS_PER_RUN = 1



def get_secret_or_env(name: str) -> Optional[str]:
    """Devuelve secretos priorizando variables de entorno y luego st.secrets."""
    try:
        return os.environ.get(name)
    except (AttributeError, KeyError):
        return st.secrets[name]

# Cliente Tavily
tavily_client = TavilyClient(api_key=get_secret_or_env("TAVILY_API_KEY")) if get_secret_or_env("TAVILY_API_KEY") else None

def tavily_search(**kwargs):
    if not tavily_client:
        raise RuntimeError("Tavily API key is not configured.")
    results = tavily_client.search(**kwargs)
    return results

tools = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Search the web with Tavily for up-to-date information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    }
]

def build_completion_kwargs(messages: List[dict], llm_config: dict, include_tools: bool = True) -> dict:
    """Prepara los argumentos para litellm según el proveedor seleccionado."""
    params = {
        "model": llm_config["model"],
        "messages": messages,
        "temperature": llm_config["temperature"],
    }
    if include_tools and tavily_client:
        params["tools"] = tools
    if llm_config.get("api_base"):
        params["api_base"] = llm_config["api_base"]
    if llm_config.get("api_key"):
        params["api_key"] = llm_config["api_key"]

    provider = llm_config.get("provider")
    if provider == "ollama":
        if llm_config.get("top_k") is not None:
            params["top_k"] = llm_config["top_k"]
        if llm_config.get("top_p") is not None:
            params["top_p"] = llm_config["top_p"]
        if llm_config.get("seed") is not None:
            params["seed"] = llm_config["seed"]
    else:
        if llm_config.get("top_p") is not None:
            params["top_p"] = llm_config["top_p"]

    return params


@st.cache_resource
def load_embedder(model_name: str, device: str):
    """
    Intenta cargar el modelo de embeddings por id (Hugging Face). Si falla, intenta cargar desde carpetas locales conocidas
    (p. ej. 'Qwen3-0.6B') y devuelve una tupla (embedder, ruta_usada).
    """
    import os
    tried = []
    # intento 1: modelo remoto/identificador
    try:
        emb = SentenceTransformer(model_name, device=device)
        return emb, model_name
    except Exception as e:
        tried.append(f"{model_name}: {e}")

    # intento 2: buscar carpeta local en workspace
    local_candidates = [
        os.path.join(os.getcwd(), "Qwen3-0.6B"),
        os.path.join(os.getcwd(), "Qwen3", "Qwen3-Embedding-0.6B"),
        "Qwen3-0.6B",
    ]
    for cand in local_candidates:
        if os.path.exists(cand):
            try:
                emb = SentenceTransformer(cand, device=device)
                return emb, cand
            except Exception as e:
                tried.append(f"{cand}: {e}")

    # si llegamos aquí, no se pudo cargar nada
    raise RuntimeError("No se pudo cargar el modelo de embeddings. Intentos: " + " | ".join(tried))


def embed_text(embedder, text: str):
    """ Devuelve el embedding del texto como una lista de floats. """
    return embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """ Extrae el texto de cada página del PDF y devuelve una lista de tuplas (número_de_página, texto). """
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            pages.append((i + 1, txt))
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """ Divide el texto en chunks de tamaño chunk_size con overlap opcional. """
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for w in words:
        current.append(w)
        current_len += len(w) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            if overlap > 0:
                ov_words = []
                ov_len = 0
                for tok in reversed(current):
                    ov_words.insert(0, tok)
                    ov_len += len(tok) + 1
                    if ov_len >= overlap:
                        break
                current = ov_words.copy()
                current_len = sum(len(tok) + 1 for tok in current)
            else:
                current = []
                current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def ingest_pdf_to_chromadb(pdf_path: str, collection, embedder, chunk_size: int = 1000, overlap: int = 200):
    """ Ingesta un PDF en una colección de ChromaDB, dividiendo el texto en chunks y almacenando embeddings. """
    pages = extract_text_from_pdf(pdf_path)
    total = 0
    for page_number, page_text in pages:
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            emb = embed_text(embedder, chunk)
            doc_id = str(uuid.uuid4())
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                ids=[doc_id],
                metadatas=[{"source": pdf_path, "page": page_number}]
            )
            total += 1
    return total


def retrieve_context(collection, embedder, query: str, n_results: int = 3) -> str:
    """ Recupera el contexto relevante de la colección para la consulta dada. """
    query_emb = embed_text(embedder, query)
    results = collection.query(query_embeddings=[query_emb], n_results=n_results)
    docs = results.get('documents', [])
    if docs and len(docs) > 0:
        # tomar el primer conjunto de documentos (para la primera consulta)
        first = docs[0]
        if isinstance(first, list):
            return " ".join(first)
        return str(first)
    return ""


def execute_tool_call(tool_call: dict) -> Tuple[str, List[dict]]:
    """Ejecuta la función solicitada por el modelo y devuelve contenido serializado y resultados crudos."""
    if tool_call.get("type") != "function":
        return json.dumps({"error": "Tipo de herramienta no soportado."}, ensure_ascii=False), []

    name = tool_call.get("function", {}).get("name")
    raw_args = tool_call.get("function", {}).get("arguments", "{}")
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Argumentos inválidos: {exc}"}, ensure_ascii=False), []

    if name == "tavily_search":
        result = tavily_search(**args)
        raw_results = result.get("results", []) if isinstance(result, dict) else []
        query = args.get("query", "")
        if raw_results:
            summary_lines = [
                f"Título: {item.get('title', 'Sin título')} | URL: {item.get('url', '')} | Resumen: {item.get('content') or item.get('snippet', '')}"
                for item in raw_results
            ]
            summary_text = "\n".join(summary_lines)
        else:
            detail = result.get("answer") if isinstance(result, dict) else ""
            detail = detail or result.get("error") if isinstance(result, dict) else detail
            detail = detail or "Sin resultados devueltos por Tavily."
            summary_text = f"Búsqueda sin resultados para '{query}'. Detalle: {detail}"
        return summary_text, raw_results

    return json.dumps({"error": f"Herramienta '{name}' no implementada."}, ensure_ascii=False), []


def rag_query(question: str, collection, embedder, llm_config: dict, n_results: int) -> Tuple[str, List[dict]]:
    """Ejecuta la consulta RAG y coordina tool-calling hasta obtener una respuesta final."""
    context = retrieve_context(collection, embedder, question, n_results=n_results)
    prompt = f"""
    Usa el siguiente contexto para responder la pregunta. Si necesitas información adicional o más actualizada, solicita una búsqueda web usando tavily_search.

    Contexto: {context}

    Pregunta: {question}
    """

    messages: List[dict] = [
        {"role": "system", "content": agents_md},
        {"role": "user", "content": prompt},
    ]
    collected_web_results: List[dict] = []
    tool_calls_used = 0

    # Permitimos varios ciclos en caso de que el modelo necesite múltiples búsquedas.
    for _ in range(6):
        allow_tools = tavily_client is not None and tool_calls_used < MAX_TOOL_CALLS_PER_RUN
        params = build_completion_kwargs(messages, llm_config, include_tools=allow_tools)
        response = completion(**params)
        choice = response["choices"][0]
        message = choice.get("message", {})
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if not allow_tools:
                messages.append(
                    {
                        "role": "system",
                        "content": "Ya ejecutaste la herramienta disponible. Usa la información entregada para responder sin llamar nuevas funciones.",
                    }
                )
                continue

            for tool_call in tool_calls:
                if tool_calls_used >= MAX_TOOL_CALLS_PER_RUN:
                    break
                tool_content, raw_results = execute_tool_call(tool_call)
                if raw_results:
                    collected_web_results.extend(raw_results)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": tool_content,
                    }
                )
                tool_calls_used += 1

            if tool_calls_used >= MAX_TOOL_CALLS_PER_RUN:
                messages.append(
                    {
                        "role": "system",
                        "content": "La búsqueda web ya se ejecutó una vez. Integra esos hallazgos con el contexto local y entrega tu respuesta final.",
                    }
                )
            continue

        content = (message.get("content") or "").strip()
        if content:
            return content, collected_web_results

        if choice.get("finish_reason") not in ("tool_calls", None):
            # Si llegamos aquí sin contenido, devolvemos un mensaje de error genérico.
            break

    raise RuntimeError("No fue posible obtener una respuesta final del modelo tras usar las herramientas.")

# Main Streamlit app
def main():
    # Estilos personalizados con color base #2991AB y degradados
    st.markdown(
        '''
        <style>
        body, .main, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #eaf6fa 0%, #2991AB 150%);
        }
        [data-testid="stSidebar"], .stSidebar {
            background: #eaf6fa !important;
            color: #17647A;
        }
        .stButton > button {
            background: linear-gradient(90deg, #2991AB 0%, #17647A 100%);
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 0.5em 1.2em;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(41,145,171,0.15);
            transition: background 0.2s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #17647A 0%, #2991AB 100%);
            color: #fff;
        }
        .stTextArea textarea {
            background: #eaf6fa;
            border: 1px solid #2991AB;
        }
        .stHeader, .stTitle, h1, h2, h3, h4 {
            color: #17647A;
        }
        .stSuccess, .stAlert-success {
            background: #2991AB22;
            color: #17647A;
            border-left: 5px solid #2991AB;
        }
        .stWarning, .stAlert-warning {
            background: #ffe6e6;
            color: #17647A;
            border-left: 5px solid #2991AB;
        }
        .stInfo, .stAlert-info {
            background: #eaf6fa;
            color: #17647A;
            border-left: 5px solid #2991AB;
        }
        .stMarkdown, .markdown-text-container {
            color: #17647A;
        }
        </style>
        ''', unsafe_allow_html=True
    )
    st.title("Agente Plan — AI")

    # controles en la sidebar
    st.sidebar.header("Configuración")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.write(f"Dispositivo detectado: {device}")

    provider_cache = st.session_state.setdefault("provider_cache", {})
    default_provider = st.session_state.get("llm_provider", PROVIDER_OPTIONS[0])
    provider_index = PROVIDER_OPTIONS.index(default_provider) if default_provider in PROVIDER_OPTIONS else 0
    provider_label = st.sidebar.selectbox("Proveedor de modelo", PROVIDER_OPTIONS, index=provider_index)
    st.session_state.llm_provider = provider_label

    temp = st.sidebar.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.1, step=0.1)
    top_k = st.sidebar.slider("top_k (núm. de tokens candidatos)", min_value=0, max_value=100, value=0, step=1)
    top_p = st.sidebar.slider("top_p (probabilidad de tokens acumulada)", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
    seed_input = st.sidebar.number_input("Seed (enter -1 para aleatorio/no establecido)", min_value=-1, value=-1, step=1)
    n_results = st.sidebar.number_input("Resultados a recuperar", min_value=1, max_value=10, value=3, step=1)

    if provider_label == "Ollama (local)":
        raw_model = st.sidebar.text_input("Modelo Ollama", value=provider_cache.get("ollama_model", DEFAULT_OLLAMA_MODEL))
        provider_cache["ollama_model"] = raw_model
        raw_base = st.sidebar.text_input("Base URL Ollama", value=provider_cache.get("ollama_api_base", DEFAULT_OLLAMA_BASE))
        provider_cache["ollama_api_base"] = raw_base
        model_id = raw_model.strip() or DEFAULT_OLLAMA_MODEL
        api_base = (raw_base.strip() or DEFAULT_OLLAMA_BASE).rstrip("/")
        llm_config = {
            "provider": "ollama",
            "label": provider_label,
            "model": model_id,
            "api_base": api_base,
            "api_key": None,
            "temperature": temp,
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "top_p": top_p,
            "seed": int(seed_input) if seed_input >= 0 else None,
            "requires_api_key": False,
        }
    elif provider_label == "GitHub Models":
        raw_model = st.sidebar.text_input("Modelo GitHub", value=provider_cache.get("github_model", DEFAULT_GITHUB_MODEL))
        provider_cache["github_model"] = raw_model
        raw_base = st.sidebar.text_input("API base GitHub", value=provider_cache.get("github_api_base", DEFAULT_GITHUB_BASE))
        provider_cache["github_api_base"] = raw_base
        model_id = raw_model.strip() or DEFAULT_GITHUB_MODEL
        api_base = (raw_base.strip() or DEFAULT_GITHUB_BASE).rstrip("/")
        token_env = get_secret_or_env("GITHUB_MODELS_TOKEN")
        if token_env and token_env.strip():
            token = token_env.strip()
            st.sidebar.caption("Token tomado de la variable 'GITHUB_MODELS_TOKEN'.")
        else:
            token_input = st.sidebar.text_input("Token GitHub Models", value=provider_cache.get("github_token", ""), type="password")
            provider_cache["github_token"] = token_input
            token = (token_input or "").strip()
        llm_config = {
            "provider": "github",
            "label": provider_label,
            "model": model_id,
            "api_base": api_base,
            "api_key": token,
            "temperature": temp,
            "top_k": None,
            "top_p": top_p,
            "seed": None,
            "requires_api_key": True,
        }
    else:
        raw_model = st.sidebar.text_input("Modelo Cerebras", value=provider_cache.get("cerebras_model", DEFAULT_CEREBRAS_MODEL))
        provider_cache["cerebras_model"] = raw_model
        raw_base = st.sidebar.text_input("API base Cerebras", value=provider_cache.get("cerebras_api_base", DEFAULT_CEREBRAS_BASE))
        provider_cache["cerebras_api_base"] = raw_base
        model_id = raw_model.strip() or DEFAULT_CEREBRAS_MODEL
        api_base = (raw_base.strip() or DEFAULT_CEREBRAS_BASE).rstrip("/")
        token_env = get_secret_or_env("CEREBRAS_API_KEY")
        if token_env and token_env.strip():
            token = token_env.strip()
            st.sidebar.caption("Token tomado de la variable 'CEREBRAS_API_KEY'.")
        else:
            token_input = st.sidebar.text_input("Token Cerebras", value=provider_cache.get("cerebras_token", ""), type="password")
            provider_cache["cerebras_token"] = token_input
            token = (token_input or "").strip()
        llm_config = {
            "provider": "cerebras",
            "label": provider_label,
            "model": model_id,
            "api_base": api_base,
            "api_key": token,
            "temperature": temp,
            "top_k": None,
            "top_p": top_p,
            "seed": None,
            "requires_api_key": True,
        }

    st.session_state.llm_config = llm_config

    st.sidebar.markdown(f"**Modelo activo:** {llm_config['model']}")
    if llm_config.get("api_base"):
        st.sidebar.caption(f"Endpoint: {llm_config['api_base']}")
    if llm_config.get("requires_api_key") and not llm_config.get("api_key"):
        st.sidebar.warning("Este proveedor requiere un token válido.")
    if llm_config.get("provider") != "ollama":
        st.sidebar.caption("`top_k` solo aplica cuando usas Ollama.")

    uploaded = st.file_uploader("Sube un PDF para ingestar", type=["pdf"])

    if 'client' not in st.session_state:
        st.session_state.client = None
        st.session_state.collection = None
        st.session_state.embedder = None
        st.session_state.embedder_source = None
        st.session_state.ready = False
        st.session_state.collection_name = None

    if uploaded is not None:
        with st.spinner("Guardando archivo temporal..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp_path = tmp.name
        st.success(f"Archivo guardado en {tmp_path}")

        if st.button("Ingestar PDF en ChromaDB"):
            with st.spinner("Cargando modelo de embeddings..."):
                embedder, embedder_source = load_embedder(EMBED_MODEL_NAME, device)
                st.session_state.embedder = embedder
                st.session_state.embedder_source = embedder_source
            client = chromadb.Client()
            collection_name = f"docs_{uuid.uuid4().hex}"
            collection = client.get_or_create_collection(name=collection_name)
            st.session_state.client = client
            st.session_state.collection = collection
            st.session_state.collection_name = collection_name
            st.session_state.ready = True
            with st.spinner("Extrayendo e ingestado... Esto puede tardar según el PDF y el modelo"):
                cnt = ingest_pdf_to_chromadb(tmp_path, collection, embedder, chunk_size=1000, overlap=200)
            st.success(f"Ingestados {cnt} chunks en la colección '{collection_name}'")

    st.markdown("---")
    st.header("Dar instrucción al modelo")
    question = st.text_area("Escribe lo que necesitas acá", height=120)

    # Mostrar la fuente del embedder si está disponible
    if st.session_state.get('embedder_source'):
        st.sidebar.info(f"Embedder cargado desde: {st.session_state.embedder_source}")

    if st.button("Enviar solicitud"):
        llm_config = st.session_state.get("llm_config")
        if not question.strip():
            st.warning("Escribe una pregunta antes de enviar.")
        elif not st.session_state.get('ready'):
            st.warning("No hay ninguna colección ingested. Sube e ingesta un PDF primero.")
        elif not llm_config:
            st.error("No se encontró la configuración del modelo. Actualiza la selección en la barra lateral.")
        elif llm_config.get("requires_api_key") and not llm_config.get("api_key"):
            st.warning("Configura un token válido para el proveedor seleccionado antes de preguntar.")
        else:
            placeholder = st.empty()
            try:
                with st.spinner("Solicitando respuesta al modelo..."):
                    answer, web_results = rag_query(
                        question,
                        st.session_state.collection,
                        st.session_state.embedder,
                        llm_config,
                        n_results,
                    )
                placeholder.markdown(answer or "_Sin texto devuelto por el modelo._")
                st.success("Respuesta completa")
                if web_results:
                    with st.expander("Resultados web consultados"):
                        for item in web_results:
                            title = item.get("title") or "Resultado sin título"
                            url = item.get("url") or ""
                            snippet = item.get("content") or item.get("snippet") or ""
                            st.markdown(f"- [{title}]({url})\n\n{snippet}")
            except Exception as exc:
                st.error(f"Error al solicitar respuesta: {exc}")

    # Mostrar estado de la colección en la sidebar
    if st.session_state.get('ready'):
        st.sidebar.success(f"Colección lista: {st.session_state.get('collection_name')}")
    else:
        st.sidebar.info("No hay colección ingested aún.")


if __name__ == "__main__":
    main()
