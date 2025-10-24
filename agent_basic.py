import os
import json
import time
from typing import List
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

assistente_nome = "Pascal"
model_name = "gpt-4o"

load_dotenv() 

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("PENAI_API_KEY não encontrada no arquivo .env")

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY não encontrada no arquivo .env")

search = GoogleSerperAPIWrapper()

def load_system_prompt(template_name: str = "system_prompt.j2", **kwargs) -> str:
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    return template.render(**kwargs)

@tool
def buscar_na_web(query: str) -> str:
    """
    Busca informações na web usando Google Serper API.
    
    Args:
        query: Termo de busca
    
    Returns:
        Resultados da busca
    """
    print(f"🔍 Buscando na web: {query}")
    
    try:
        search = GoogleSerperAPIWrapper(gl='pt', hl='br')
        resultados = search.run(query)

        return resultados
    except Exception as e:
        print(f"❌ Erro ao buscar na web: {e}")
        return f"Erro ao buscar: {str(e)}"

TOOLS = {t.name: t for t in [buscar_na_web]}

llm = ChatOpenAI(model=model_name, temperature=0.2)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

def chat_loop():
    system_prompt = load_system_prompt(
        assistente_nome=assistente_nome,
        data_atual=time.strftime("%d/%m/%Y %H:%M:%S")
    )
    
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt)
    ]
    print(f"{assistente_nome}: Olá! (digite 'sair' para encerrar)\n")

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            print(f"{assistente_nome}: Até logo!")
            break

        messages.append(HumanMessage(content=user_input))

        while True:
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            tool_calls = ai_msg.tool_calls
            
            if not tool_calls:
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

        print(f"{assistente_nome}: {ai_msg.content}\n")


if __name__ == "__main__":
    chat_loop()