import os
from pathlib import Path
from typing import Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from datetime import datetime

chroma_client = chromadb.PersistentClient(path="./chroma_db")

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