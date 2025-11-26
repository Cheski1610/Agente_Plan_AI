## Proyecto Agente Programa - AI

<div align="center">
    <img src="Images/Imagen_1.png" alt="Portada" style="max-width:800px; height:auto;" />
</div>

Aplicación Streamlit especializada en **redactar programas de trabajo para auditorías individuales**. Carga documentación interna (políticas, lineamientos, manuales) en forma de PDF, genera un índice vectorial en ChromaDB y consulta dicha base junto con instrucciones del archivo `agents_es.md`, que describe el flujo completo para diseñar programas de auditoría: objetivos, alcance, criterios, matriz de riesgos, pruebas y cronograma. Cuando el modelo necesita información externa (regulación reciente, riesgos emergentes), complementa el análisis con una búsqueda web mediante Tavily. Cada instrucción limita el uso de la herramienta a una sola llamada para mantener controlado el flujo y los costos.

### Características principales
- **Enfoque auditoría interna**: sigue las directrices de `agents_es.md` para construir planes de trabajo completos (objetivos, criterios, matriz de riesgos, procedimientos, cronograma) alineados con normas del IIA e ISO 31000.
- **Carga e ingesta de PDFs**: divide el documento en fragmentos, calcula embeddings con `sentence-transformers` y los almacena en ChromaDB.
- **RAG flexible**: cada consulta recupera los fragmentos más relevantes y construye un prompt combinado para el modelo seleccionado (Ollama local, GitHub Models o Cerebras).
- **Búsqueda web opcional**: si el modelo detecta que necesita información externa, puede llamar a `tavily_search`; el resultado se resume y se muestra en un expander para transparencia.
- **UI amigable**: controles en la barra lateral para hiperparámetros (`temperature`, `top_k`, `top_p`, `seed`), selección del proveedor y estado de la colección ingestada.

### Requisitos
- Python 3.12+.
- Dependencias listadas en `requirements.txt` (incluye Streamlit, ChromaDB, PyTorch con CUDA 12.6, Tavily, etc.).
- Modelos/servicios externos según el proveedor elegido:
  - Ollama corriendo de forma local si usas el modo "Ollama (local)".
  - Token para GitHub Models (`GITHUB_MODELS_TOKEN`) o Cerebras (`CEREBRAS_API_KEY`) cuando corresponda.
- API key de Tavily (`TAVILY_API_KEY`) para habilitar la búsqueda web.

### Configuración
1. Crea y activa tu entorno virtual (opcional pero recomendado).
2. Instala las dependencias:
	```bash
	pip install -r requirements.txt
	```
3. Crea un archivo `.env` en la raíz del proyecto y define, al menos:
	```env
	TAVILY_API_KEY=tu_token
	# Opcionales según el proveedor
	OLLAMA_MODEL_ID=ollama/Qwen3:8B
	OLLAMA_API_BASE=http://localhost:11434
	GITHUB_MODELS_TOKEN=ghp_xxx
	CEREBRAS_API_KEY=csk_xxx
	```
	También puedes suministrar estas variables mediante `st.secrets` si despliegas en Streamlit Cloud.

### Ejecución
1. Ejecuta la app:
	```bash
	streamlit run app_ai.py
	```
2. En la interfaz:
	- Sube un PDF y pulsa **Ingestar PDF en ChromaDB** para crear la colección.
	- Ajusta los parámetros del modelo en la barra lateral.
	- Redacta una pregunta y presiona **Enviar solicitud**.
	- Si el modelo invoca Tavily, verás un expander con los enlaces utilizados; la respuesta combinará el contexto documental y la búsqueda web.

### Notas adicionales
- `agents_es.md` define el propósito auditor (recuperar contexto interno, solicitar información faltante, integrar riesgos externos y producir el entregable en español). Modifica ese archivo para adaptar el enfoque a tu metodología.
- El agente limita el uso de herramientas a una llamada por ejecución para evitar bucles o costos excesivos.
- Si PyTorch no encuentra una rueda compatible (especialmente en GPU/Windows), sigue las instrucciones oficiales de [pytorch.org](https://pytorch.org/get-started/locally/).