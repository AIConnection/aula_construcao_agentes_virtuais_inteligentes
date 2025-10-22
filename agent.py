import os
import math
import re
from typing import List

from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from groq import BadRequestError

# ==========================
# Configuração
# ==========================
os.environ["GROQ_API_KEY"] = 'YOUR_GROQ_API_KEY_HERE'  # Substitua pela sua chave de API Groq

def make_llm(model_name: str = "openai/gpt-oss-120b") -> ChatGroq:
    return ChatGroq(
        model=model_name,
        temperature=0.3,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

# ==========================
# Ferramentas reais
# ==========================
def search_web(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "Nenhum resultado encontrado."
        return "Principais resultados:\n" + "\n".join(
            f"- {r.get('title','(sem título)')} ({r.get('href','')})" for r in results
        )
    except Exception as e:
        return f"Erro ao buscar: {e}"

def calculate_expression(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Resultado: {result}"
    except Exception as e:
        return f"Erro ao calcular: {e}"

def detect_tool(user_input: str):
    low = user_input.lower()
    if any(w in low for w in ["buscar", "procure", "pesquise", "pesquisar"]):
        return "buscar"
    if any(w in low for w in ["calcule", "calcular", "resultado de", "quanto é"]):
        return "calcular"
    return None

def extract_search_query(user_input: str) -> str:
    q = re.sub(r"^(buscar|procure|pesquise|pesquisar)\s*(por)?\s*", "", user_input, flags=re.I).strip()
    return q if q else user_input

def extract_expression(user_input: str) -> str:
    m = re.search(r"(calcule|calcular|resultado de|quanto é)\s*(.*)", user_input, flags=re.I)
    return (m.group(2).strip() if m and m.group(2).strip() else user_input).strip()

# ==========================
# Chat loop
# ==========================
def chat_loop():
    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                "Você é um assistente útil que ajuda o usuário a aprender programação, "
                "respondendo de forma clara e concisa, sempre em português. "
                "Quando o usuário pedir busca ou cálculo, responda usando as ferramentas."
            )
        )
    ]
    print("Assistente: Olá! (digite 'sair' para encerrar)\n")
    llm = make_llm()
    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Assistente: Até logo!")
            break

        tool = detect_tool(user_input)
        if tool == "buscar":
            query = extract_search_query(user_input)
            print("Assistente (ferramenta): pesquisando...\n")
            print(search_web(query) + "\n")
            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content="(resultado retornado pela ferramenta de busca)"))
            continue

        if tool == "calcular":
            expr = extract_expression(user_input)
            print("Assistente (ferramenta): " + calculate_expression(expr) + "\n")
            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content="(resultado retornado pela ferramenta de cálculo)"))
            continue

        # Conversa normal via LLM com fallback de modelo se deprecado
        messages.append(HumanMessage(content=user_input))
        try: 
            ai = llm.invoke(messages)  # retorna AIMessage
        except BadRequestError as e:
            # fallback automático para um modelo suportado atual
            print("[Aviso] Modelo atual indisponível/deprecado. Alternando para 'openai/gpt-oss-120b'...\n")
            llm = make_llm("openai/gpt-oss-120b")
            ai = llm.invoke(messages)
        content = ai.content or ""
        print(f"Assistente: {content}\n")
        messages.append(AIMessage(content=content))

if __name__ == "__main__":
    chat_loop()