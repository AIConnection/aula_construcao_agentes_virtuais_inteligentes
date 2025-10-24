import os
import json
import time
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import chromadb
from chromadb.config import Settings
import hashlib
from datetime import datetime, timedelta

assistente_nome = "Pascal"
model_name = "gpt-4o"

load_dotenv() 

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY não encontrada no arquivo .env")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="qa_history",
    metadata={"hnsw:space": "cosine"}
)

embeddings = OpenAIEmbeddings()

def salvar_qa_no_chroma(pergunta: str, resposta: str, fonte: str = "web", ttl_dias: int = 7) -> None:
    """
    Salva par pergunta-resposta no ChromaDB com timestamp.
    
    Args:
        pergunta: Pergunta do usuário
        resposta: Resposta fornecida
        fonte: Origem da resposta (assistant, web_search)
        ttl_dias: Tempo de vida do cache em dias
    """
    doc_id = hashlib.md5(f"{pergunta}{time.time()}".encode()).hexdigest()
    embedding = embeddings.embed_query(pergunta)
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[resposta],
        metadatas=[{
            "pergunta": pergunta,
            "fonte": fonte,
            "timestamp": datetime.now().isoformat(),
            "data_validade": (datetime.now() + timedelta(days=ttl_dias)).isoformat()
        }]
    )
    print(f"💾 Resposta salva no ChromaDB (fonte: {fonte}, válida por {ttl_dias} dias)")

def buscar_no_chroma(pergunta: str, threshold: float = 0.7) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Busca respostas similares no ChromaDB.
    
    Args:
        pergunta: Pergunta para buscar
        threshold: Limiar de similaridade (0-1)
    
    Returns:
        Tupla (resposta, metadados) ou (None, None) se não encontrar
    """
    if collection.count() == 0:
        return None, None
    
    embedding = embeddings.embed_query(pergunta)
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    
    if not results["documents"] or not results["documents"][0]:
        return None, None
    
    distancia = 1.0
    if results["distances"] and results["distances"][0]:
        distancia = float(results["distances"][0][0])
    
    similaridade = 1 - distancia
    
    if not results["metadatas"] or not results["metadatas"][0]:
        return None, None
    
    metadata_raw = results["metadatas"][0][0]
    
    metadata: Dict[str, Any] = {
        "pergunta": str(metadata_raw.get("pergunta", "")),
        "fonte": str(metadata_raw.get("fonte", "")),
        "timestamp": str(metadata_raw.get("timestamp", "")),
        "data_validade": str(metadata_raw.get("data_validade", ""))
    }
    
    data_validade_str = metadata.get("data_validade", "")
    if data_validade_str:
        try:
            data_validade = datetime.fromisoformat(data_validade_str)
            if datetime.now() > data_validade:
                print(f"⏰ Cache expirado para esta consulta")
                return None, None
        except (ValueError, TypeError):
            print(f"⚠️ Data de validade inválida no cache")
            return None, None
    
    if similaridade >= threshold:
        print(f"✅ Resposta encontrada no ChromaDB (similaridade: {similaridade:.2%})")
        documento = str(results["documents"][0][0])
        return documento, metadata
    
    return None, None

search = GoogleSerperAPIWrapper()

def load_system_prompt(template_name: str = "system_prompt_agent_web_cromadb.j2", **kwargs) -> str:
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    return template.render(**kwargs)

@tool
def buscar_na_web(query: str) -> str:
    """
    Busca informações atualizadas na web usando Google Serper API.
    
    IMPORTANTE: Esta ferramenta SEMPRE faz uma busca real na web, sem usar cache.
    Use quando precisar de:
    - Informações em tempo real ou muito recentes
    - Notícias atuais e eventos recentes
    - Preços, cotações ou dados que mudam frequentemente
    - Qualquer informação que pode ter mudado desde a última consulta
    
    Os resultados serão automaticamente armazenados para consultas futuras.
    
    Args:
        query: Termo de busca para pesquisar na web
    
    Returns:
        Resultados atualizados da busca na web
    """
    print(f"🌐 Buscando na web: {query}")
    
    try:
        search = GoogleSerperAPIWrapper()
        resultados = search.run(query)
        resultados_str = str(resultados) if resultados else "Nenhum resultado encontrado"
        
        # Armazena o resultado da busca com TTL curto (1 dia para dados web)
        salvar_qa_no_chroma(query, resultados_str, fonte="web_search", ttl_dias=1)
        
        print(f"✅ Resultados obtidos e armazenados")
        return resultados_str
    except Exception as e:
        erro_msg = f"Erro ao buscar: {str(e)}"
        print(f"❌ {erro_msg}")
        return erro_msg

@tool
def consultar_conhecimento(pergunta: str) -> str:
    """
    Consulta conhecimento armazenado no histórico de conversas (ChromaDB).
    
    Esta ferramenta busca em respostas anteriores e conhecimento já processado.
    Use quando:
    - A informação é estável e não muda com frequência
    - Você quer verificar se já respondemos algo similar antes
    - Precisa de conceitos, definições ou conhecimento geral
    - A pergunta não requer dados atualizados
    
    Se não encontrar aqui, você pode usar buscar_na_web para obter informações frescas.
    
    Args:
        pergunta: Pergunta para buscar no conhecimento armazenado
    
    Returns:
        Resposta encontrada ou mensagem indicando que não foi encontrada
    """
    print(f"📚 Consultando conhecimento armazenado: {pergunta}")
    resposta, metadata = buscar_no_chroma(pergunta, threshold=0.75)
    
    if resposta and metadata:
        timestamp = metadata.get("timestamp", "data desconhecida")
        fonte = metadata.get("fonte", "desconhecida")
        
        # Calcula há quanto tempo foi armazenado
        try:
            data_armazenamento = datetime.fromisoformat(timestamp)
            tempo_decorrido = datetime.now() - data_armazenamento
            if tempo_decorrido.days > 0:
                tempo_str = f"há {tempo_decorrido.days} dia(s)"
            elif tempo_decorrido.seconds > 3600:
                tempo_str = f"há {tempo_decorrido.seconds // 3600} hora(s)"
            else:
                tempo_str = f"há {tempo_decorrido.seconds // 60} minuto(s)"
        except:
            tempo_str = "em data desconhecida"
        
        return f"✓ Informação encontrada (armazenada {tempo_str}):\n\n{resposta}"
    else:
        return "✗ Não encontrei informações sobre isso no conhecimento armazenado. Considere usar buscar_na_web para obter dados atualizados."

TOOLS = {t.name: t for t in [buscar_na_web, consultar_conhecimento]}

llm = ChatOpenAI(model=model_name, temperature=0.2)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

def chat_loop() -> None:
    system_prompt = load_system_prompt(
        assistente_nome=assistente_nome,
        data_atual=time.strftime("%d/%m/%Y %H:%M:%S")
    )
    
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt)
    ]
    print(f"{assistente_nome}: Olá! Sou seu assistente pesssoal. (digite 'sair' para encerrar)\n")

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            print(f"{assistente_nome}: Até logo!")
            break

        messages.append(HumanMessage(content=user_input))
        pergunta_usuario = user_input

        while True:
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            tool_calls = ai_msg.tool_calls
            
            if not tool_calls:
                if ai_msg.content and isinstance(ai_msg.content, str):
                    salvar_qa_no_chroma(pergunta_usuario, ai_msg.content, fonte="assistant", ttl_dias=30)
                break

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                tool = TOOLS.get(tool_name)
                if not tool:
                    result = f"Ferramenta '{tool_name}' não encontrada."
                else:
                    try:
                        result = tool.invoke(tool_args)
                    except Exception as e:
                        result = f"Erro ao executar {tool_name}: {e}"

                messages.append(
                    ToolMessage(
                        name=tool_name,
                        content=str(result),
                        tool_call_id=tool_id,
                    )
                )

        print(f"\n{assistente_nome}: {ai_msg.content}\n")


if __name__ == "__main__":
    chat_loop()