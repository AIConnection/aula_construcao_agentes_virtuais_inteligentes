import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# ==========================================================
# Configuração
# ==========================================================
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"  # Substitua pela sua chave de API
model_name = "gpt-4o"

# ==========================================================
# Tools
# ==========================================================
@tool
def somar(a: float, b: float) -> float:
    """Soma dois números."""
    print(f"⚙️ Executando somar({a}, {b})")
    return a + b

@tool
def inverter(texto: str) -> str:
    """Inverte o texto fornecido."""
    print(f"⚙️ Executando inverter('{texto}')")
    return texto[::-1]

TOOLS = {t.name: t for t in [somar, inverter]}

# ==========================================================
# Modelo com suporte a function calling
# ==========================================================
llm = ChatOpenAI(model=model_name, temperature=0.2)
llm_with_tools = llm.bind_tools(list(TOOLS.values()))

# ==========================================================
# Chat Loop
# ==========================================================
def chat_loop():
    messages = [
        SystemMessage(content="Você é um assistente útil que usa ferramentas quando necessário.")
    ]
    print("Assistente: Olá! (digite 'sair' para encerrar)\n")

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Assistente: Até logo!")
            break

        messages.append(HumanMessage(content=user_input))

        # Loop para lidar com múltiplas chamadas de ferramentas
        while True:
            # 1️⃣ Invocar o modelo
            ai_msg: AIMessage = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # 2️⃣ Verificar se há tool_calls
            tool_calls = ai_msg.tool_calls
            
            if not tool_calls:
                # Sem ferramentas, apenas resposta final
                break

            # 3️⃣ Executar cada ferramenta chamada
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                tool = TOOLS.get(tool_name)
                if not tool:
                    result = f"❌ Ferramenta '{tool_name}' não encontrada."
                else:
                    try:
                        result = tool.invoke(tool_args)
                    except Exception as e:
                        result = f"Erro ao executar {tool_name}: {e}"

                # 4️⃣ Adicionar resposta da ferramenta às mensagens
                messages.append(
                    ToolMessage(
                        name=tool_name,
                        content=str(result),
                        tool_call_id=tool_id,
                    )
                )

            # O loop continua para permitir que o modelo processe os resultados

        print(f"Assistente: {ai_msg.content}\n")


if __name__ == "__main__":
    chat_loop()