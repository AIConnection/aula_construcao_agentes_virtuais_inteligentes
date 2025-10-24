import os
import json
import time
import sys
from pathlib import Path
import asyncio
from pydantic.types import SecretStr
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_sandbox import PyodideSandbox
from langchain_community.tools import ShellTool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import hashlib
from datetime import datetime, timedelta

load_dotenv() 

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY não encontrada no arquivo .env")

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY não encontrada no arquivo .env")

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
assistent_name = os.getenv("ASSISTENT_NAME", "Pascal")
system_prompt = os.getenv("SYSTEM_PROMPT_TEMPLATE", "system_prompt.j2")

llm = ChatOpenAI(
    base_url=base_url,
    model=model_name,
    temperature=0.2)

search = GoogleSerperAPIWrapper()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection_qa = chroma_client.get_or_create_collection(
    name="qa_history",
    metadata={"hnsw:space": "cosine"}
)

collection_docs = chroma_client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

embeddings = OpenAIEmbeddings()

sandbox = PyodideSandbox(
    stateful=True,
    allow_env=True,
    allow_read=True,
    allow_write=True,
    allow_run=True,
    allow_net=True
)

shell_tool = ShellTool()


def processar_pdf(caminho_pdf: str) -> Dict[str, Any]:
    """
    Processa um PDF e armazena chunks no ChromaDB.
    """
    try:
        print(f"\n📄 Processando: {caminho_pdf}")
        
        # Verifica se arquivo existe
        if not os.path.exists(caminho_pdf):
            return {"sucesso": False, "erro": f"Arquivo não encontrado: {caminho_pdf}"}
        
        # Carrega o PDF
        loader = PyMuPDFLoader(caminho_pdf)
        documents = loader.load()
        
        print(f"   📊 {len(documents)} página(s) carregada(s)")
        
        # Divide em chunks
        chunks = text_splitter.split_documents(documents)
        print(f"   ✂️  Dividido em {len(chunks)} chunks")
        
        # Prepara dados
        ids = []
        embeddings_list = []
        documents_list = []
        metadatas_list = []
        
        nome_arquivo = os.path.basename(caminho_pdf)
        
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{nome_arquivo}_{i}_{chunk.page_content[:100]}".encode()
            ).hexdigest()
            ids.append(chunk_id)
            
            # Gera embedding
            embedding = embeddings.embed_query(chunk.page_content)
            embeddings_list.append(embedding)
            
            documents_list.append(chunk.page_content)
            
            metadata = {
                "arquivo": nome_arquivo,
                "caminho": caminho_pdf,
                "pagina": chunk.metadata.get("page", i),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                "tipo": "pdf_chunk"
            }
            metadatas_list.append(metadata)
            
            # Mostra progresso
            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                print(f"   🔄 Processando embeddings: {i + 1}/{len(chunks)}", end='\r')
        
        print()  # Nova linha após progresso
        
        # Insere no ChromaDB
        collection_docs.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents_list,
            metadatas=metadatas_list
        )
        
        print(f"   ✅ Armazenado com sucesso!")
        
        return {
            "sucesso": True,
            "arquivo": nome_arquivo,
            "paginas": len(documents),
            "chunks": len(chunks)
        }
        
    except Exception as e:
        erro_msg = f"Erro ao processar: {str(e)}"
        print(f"   ❌ {erro_msg}")
        return {"sucesso": False, "erro": erro_msg}

def processar_diretorio(caminho_dir: str, recursivo: bool = False) -> Dict[str, Any]:
    """
    Processa todos os PDFs em um diretório.
    """
    caminho = Path(caminho_dir)
    
    if not caminho.exists():
        return {"sucesso": False, "erro": f"Diretório não encontrado: {caminho_dir}"}
    
    # Busca PDFs
    if recursivo:
        pdfs = list(caminho.rglob("*.pdf"))
    else:
        pdfs = list(caminho.glob("*.pdf"))
    
    if not pdfs:
        return {"sucesso": False, "erro": "Nenhum PDF encontrado no diretório"}
    
    print(f"\n📁 Encontrados {len(pdfs)} arquivo(s) PDF")
    print("=" * 60)
    
    resultados = {
        "total": len(pdfs),
        "sucesso": 0,
        "falha": 0,
        "detalhes": []
    }
    
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}]", end=" ")
        resultado = processar_pdf(str(pdf))
        
        if resultado["sucesso"]:
            resultados["sucesso"] += 1
        else:
            resultados["falha"] += 1
        
        resultados["detalhes"].append({
            "arquivo": str(pdf),
            "resultado": resultado
        })
    
    return resultados

def listar_documentos():
    """
    Lista todos os documentos armazenados no ChromaDB.
    """
    total = collection_docs.count()
    
    if total == 0:
        print("\n📭 Nenhum documento armazenado.")
        return
    
    # Busca todos os metadados únicos por arquivo
    results = collection_docs.get()
    
    arquivos = {}
    for metadata in results["metadatas"] or []:
        arquivo = metadata.get("arquivo", "desconhecido")
        if arquivo not in arquivos:
            arquivos[arquivo] = {
                "chunks": 0,
                "paginas": set(),
                "timestamp": metadata.get("timestamp", "")
            }
        arquivos[arquivo]["chunks"] += 1
        arquivos[arquivo]["paginas"].add(metadata.get("pagina", 0))
    
    print(f"\n📚 Documentos armazenados ({total} chunks total):")
    print("=" * 60)
    
    for i, (arquivo, info) in enumerate(arquivos.items(), 1):
        print(f"\n{i}. {arquivo}")
        print(f"   📄 Páginas: {len(info['paginas'])}")
        print(f"   🧩 Chunks: {info['chunks']}")
        print(f"   🕐 Adicionado: {info['timestamp'][:19]}")

def limpar_base():
    """
    Limpa todos os documentos do ChromaDB.
    """
    total = collection_docs.count()
    
    if total == 0:
        print("\n📭 Base já está vazia.")
        return
    
    print(f"\n⚠️  ATENÇÃO: Isso irá remover {total} chunks da base de dados.")
    confirmacao = input("Digite 'CONFIRMAR' para prosseguir: ")
    
    if confirmacao != "CONFIRMAR":
        print("❌ Operação cancelada.")
        return
    
    # Limpa a collection
    ids = collection_docs.get()["ids"]
    collection_docs.delete(ids=ids)
    
    print(f"✅ Base limpa com sucesso! {total} chunks removidos.")

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
    
    collection_qa.add(
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
    if collection_qa.count() == 0:
        return None, None
    
    embedding = embeddings.embed_query(pergunta)
    
    results = collection_qa.query(
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

def buscar_em_documentos(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    if collection_docs.count() == 0:
        return []
    
    embedding = embeddings.embed_query(query)
    results = collection_docs.query(query_embeddings=[embedding], n_results=n_results)
    
    if not results["documents"] or not results["documents"][0]:
        return []
    
    resultados_formatados = []
    for i in range(len(results["documents"][0])):
        distancia = float(results["distances"][0][i]) if results["distances"] else 1.0
        similaridade = 1 - distancia
        metadatas = results["metadatas"] or []
        metadata = metadatas[0][i]
        
        resultado = {
            "conteudo": results["documents"][0][i],
            "similaridade": similaridade,
            "arquivo": metadata.get("arquivo", "desconhecido"),
            "pagina": metadata.get("pagina", "?"),
            "chunk_index": metadata.get("chunk_index", 0)
        }
        resultados_formatados.append(resultado)
    
    return resultados_formatados

def load_system_prompt(template_name: str = system_prompt, **kwargs) -> str:
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
def consultar_memoria(pergunta: str) -> str:
    """
    Consulta memoria armazenado no histórico de conversas (ChromaDB).
    
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

@tool
def consultar_conhecimento(pergunta: str, num_resultados: int = 10) -> str:
    """
    Consulta conhecimento em documentos PDF previamente carregados.
    
    Use esta ferramenta para encontrar informações específicas em documentos.
    
    Args:
        pergunta: Pergunta ou termo de busca
        num_resultados: Número de trechos relevantes a retornar (padrão: 3)
    
    Returns:
        Trechos relevantes dos documentos com referências
    """

    print(f"📄 Consultando documentos para: {pergunta}")
    if collection_docs.count() == 0:
        return "❌ Não há documentos carregados. Execute o script em modo CARGA primeiro."
    
    resultados = buscar_em_documentos(pergunta, n_results=num_resultados)
    
    if not resultados:
        return "❌ Não encontrei informações relevantes nos documentos."
    
    resposta = f"📚 Encontrei {len(resultados)} trecho(s) relevante(s):\n\n"
    
    for i, resultado in enumerate(resultados, 1):
        resposta += f"--- Trecho {i} ---\n"
        resposta += f"📄 Arquivo: {resultado['arquivo']}\n"
        resposta += f"📄 Página: {resultado['pagina']}\n"
        resposta += f"🎯 Relevância: {resultado['similaridade']:.1%}\n\n"
        resposta += f"{resultado['conteudo']}\n\n"
    
    return resposta

@tool
def executar_codigo_python(codigo: str) -> str:
    """
    Executa código Python em um sandbox seguro usando Pyodide.
    
    Use esta ferramenta para cálculos, manipulação de dados ou qualquer tarefa que requeira execução de código.
    
    Args:
        codigo: Código Python a ser executado
    Returns:
        Resultado da execução ou mensagem de erro
    """
    print(f"💻 Executando código Python no sandbox")
    try:
        resultado = asyncio.run(sandbox.execute(codigo))
        return f"Resultado da execução:\n{resultado}"
    except Exception as e:
        erro_msg = f"Erro ao executar código: {str(e)}"
        print(f"❌ {erro_msg}")
        return erro_msg

@tool
def executar_comando_shell(comando: str) -> str:
    """
    Executa um comando shell no ambiente local.
    
    Use esta ferramenta para tarefas que requerem interação com o sistema operacional.
    
    Args:
        comando: Comando shell a ser executado
    Returns:
        Resultado da execução ou mensagem de erro
    """
    print(f"🖥️ Executando comando shell: {comando}")
    try:
        executar_comando_shell = input("Y/N: Tem certeza que deseja executar este comando shell? ")
        if executar_comando_shell.lower() != 'y':
            return "Execução de comando shell cancelada pelo usuário."
        
        resultado = shell_tool.invoke({"commands": [comando]})
        return f"Resultado da execução:\n{resultado}"
    except Exception as e:
        erro_msg = f"Erro ao executar comando shell: {str(e)}"
        print(f"❌ {erro_msg}")
        return erro_msg

TOOLS = {
    t.name: t for t in [
        buscar_na_web,
        consultar_memoria,
        consultar_conhecimento,
        executar_codigo_python,
        executar_comando_shell
    ]
}

llm_with_tools = llm.bind_tools(list(TOOLS.values()))

def chat_loop() -> None:
    system_prompt = load_system_prompt(
        assistente_nome=assistent_name,
        data_atual=time.strftime("%d/%m/%Y %H:%M:%S")
    )
    
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt)
    ]
    print(f"{assistent_name}: Olá! Sou seu assistente pesssoal. (digite 'sair' para encerrar)\n")

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            print(f"{assistent_name}: Até logo!")
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

        print(f"\n{assistent_name}: {ai_msg.content}\n")

def menu_carga():
    print("\n" + "=" * 60)
    print("📚 MODO CARGA - GERENCIAMENTO DE DOCUMENTOS")
    print("=" * 60)
    print("\n")
    print("1. carregar PDF único")
    print("2. carregar diretório (não recursivo)")
    print("3. carregar diretório (recursivo)")
    print("4. listar documentos armazenados")
    print("5. limpar base de dados")
    print("6. mostrar menu")
    print("0. sair")

def modo_carga():
    """
    Menu interativo do modo carga.
    """
    menu_carga()

    while True:
        opcao = input("\nEscolha uma opção: ").strip()
        
        if opcao == "1":
            caminho = input("\nCaminho do PDF: ").strip()
            processar_pdf(caminho)
        elif opcao == "2":
            caminho = input("\nCaminho do diretório: ").strip()
            resultados = processar_diretorio(caminho, recursivo=False)
            if resultados.get("sucesso", 0) > 0:
                print(f"\n✅ Resumo: {resultados['sucesso']} sucesso(s), {resultados['falha']} falha(s)")
        elif opcao == "3":
            caminho = input("\nCaminho do diretório: ").strip()
            resultados = processar_diretorio(caminho, recursivo=True)
            if resultados.get("sucesso", 0) > 0:
                print(f"\n✅ Resumo: {resultados['sucesso']} sucesso(s), {resultados['falha']} falha(s)")
        elif opcao == "4":
            listar_documentos()
        elif opcao == "5":
            limpar_base()
        elif opcao == "6":
            menu_carga()
        elif opcao.lower() in {"sair", "exit", "quit"}:
            print("Até logo!")
            break
        else:
            print("\n❌ Opção inválida!")

if __name__ == "__main__":

    input_mode = input("Escolha o modo (chat/carga): ").strip().lower()

    if input_mode == "carga":
        modo_carga()
    else:
        chat_loop()