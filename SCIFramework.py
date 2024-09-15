import subprocess
from abc import ABC, abstractmethod
from groq import Groq
from typing import Dict, List, Any
import re

class Belief:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Desire:
    def __init__(self, name):
        self.name = name

class Intention:
    def __init__(self, name):
        self.name = name

class BDIModel:
    def __init__(self, knowledge_context_layer):
        self.knowledge_context_layer = knowledge_context_layer
        self.beliefs = {}  # crenças do agente
        self.desires = []  # desejos do agente
        self.intentions = []  # ações do agente

    def add_belief(self, belief):
        self.beliefs[belief.name] = belief.value
        self.knowledge_context_layer.update_knowledge(belief.name, belief.value)

    def get_belief(self, name):
        return self.beliefs.get(name)
    
    def add_desire(self, desire:Desire):
        self.desires.append(desire.name)

    def get_desire(self, desire:Desire):
        return self.desires[desire.name]
    
    def add_intention(self, intention:Intention):
        self.intentions.append(intention.name)

    def get_intention(self, intention:Intention):
        return self.intentions[intention.name]


class KnowledgeContextLayer:
    def __init__(self):
        self.knowledge_base = {}  # base de conhecimento do agente

    def update_knowledge(self, key, value):
        self.knowledge_base[key] = value

    def get_knowledge(self, key):
        return self.knowledge_base.get(key)


class BDIConfig:
    def __init__(self):
        self.beliefs_config = {}  # configuração de crenças
        self.desires_config = []  # configuração de desejos
        self.intentions_config = []  # configuração de ações

    def add_belief_config(self, name, value):
        self.beliefs_config[name] = value

    def add_desire_config(self, desire):
        self.desires_config.append(desire)

    def add_intention_config(self, intention):
        self.intentions_config.append(intention)

    def get_beliefs_config(self):
        return self.beliefs_config

    def get_desires_config(self):
        return self.desires_config

    def get_intentions_config(self):
        return self.intentions_config

class InputLayer:
    def process_input(self, user_input: str) -> Dict[str, Any]:
        return {
            "raw_input": user_input,
            "processed_input": user_input.strip().lower(),
            "tokens": user_input.split(),
            "intent": self._extract_intent(user_input),
        }

    def _extract_intent(self, user_input: str) -> str:
        if user_input.startswith("!"):
            return "command"
        elif "?" in user_input:
            return "question"
        else:
            return "statement"

class KnowledgeContextLayer:
    def __init__(self):
        self.knowledge_base = {}
        self.conversation_history = []
        self.current_context = {}

    def update_knowledge(self, key: str, value: Any):
        self.knowledge_base[key] = value

    def get_knowledge(self, key: str) -> Any:
        return self.knowledge_base.get(key)

    def add_to_history(self, message: Dict[str, str]):
        self.conversation_history.append(message)
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def update_context(self, key: str, value: Any):
        self.current_context[key] = value

    def get_context(self, key: str) -> Any:
        return self.current_context.get(key)

    def get_full_context(self) -> Dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base,
            "conversation_history": self.conversation_history,
            "current_context": self.current_context
        }

class AutoKnowledgeManager:
    def __init__(self, knowledge_context: KnowledgeContextLayer):
        self.knowledge_context = knowledge_context

    def extract_knowledge(self, user_input: str, system_response: str):
        # Extrai informações-chave da interação
        key_info = self._extract_key_info(user_input, system_response)
        
        for key, value in key_info.items():
            self.knowledge_context.update_knowledge(key, value)

    def _extract_key_info(self, user_input: str, system_response: str) -> Dict[str, str]:
        # Implementação simplificada de extração de informações-chave
        key_info = {}
        
        # Procura por definições simples (ex: "X é Y")
        definitions = re.findall(r'(\w+) (é|são) ([\w\s]+)', user_input + " " + system_response)
        for match in definitions:
            key_info[match[0].lower()] = match[2]
        
        # Procura por fatos numéricos (ex: "X tem Y anos")
        facts = re.findall(r'(\w+) tem (\d+) (\w+)', user_input + " " + system_response)
        for match in facts:
            key_info[f"{match[0].lower()}_{match[2]}"] = match[1]
        
        return key_info

class LLMClient(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass

class GroqClient(LLMClient):
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        completion = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=kwargs.get('temperature', 1),
            max_tokens=kwargs.get('max_tokens', 1024),
            top_p=kwargs.get('top_p', 1),
            stream=True,
            stop=kwargs.get('stop', None),
        )
        response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            response += content
            print(content, end="")
        print()
        return response

class ProcessingLayer:
    def __init__(self, llm_client: LLMClient, auto_knowledge_manager: AutoKnowledgeManager):
        self.llm_client = llm_client
        self.auto_knowledge_manager = auto_knowledge_manager

    def generate_response(self, input_data: Dict[str, Any], knowledge_context: KnowledgeContextLayer) -> str:
        full_context = knowledge_context.get_full_context()
        
        messages = [
            {"role": "system", "content": "Você é um assistente conversacional inteligente e amigável."},
            {"role": "user", "content": f"Contexto: {full_context}\n\nEntrada do usuário: {input_data['raw_input']}\nIntenção detectada: {input_data['intent']}"}
        ]
        
        messages.extend(knowledge_context.conversation_history)
        
        response = self.llm_client.generate_response(messages)
        
        # Atualiza o conhecimento automaticamente
        self.auto_knowledge_manager.extract_knowledge(input_data['raw_input'], response)
        
        knowledge_context.add_to_history({"role": "assistant", "content": response})
        knowledge_context.update_context("last_response", response)
        
        return response

class OutputLayer:
    def format_response(self, response: str) -> str:
        return f"SCI: {response}"

class ConfigLayer:
    def __init__(self):
        self.config = {
            "language": "pt-br",
            "max_tokens": 1024,
            "temperature": 0.7
        }

    def get_config(self, key: str) -> Any:
        return self.config.get(key)

    def set_config(self, key: str, value: Any):
        self.config[key] = value

class CommandModule:
    def __init__(self, knowledge_context: KnowledgeContextLayer):
        self.knowledge_context = knowledge_context
        self.commands = {
            "help": self.show_help,
            "execute": self.execute_python_file,
            "knowledge": self.show_knowledge,
        }

    def process_command(self, command: str) -> str:
        parts = command.split(maxsplit=1)
        cmd = parts[0][1:]
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            return f"Comando desconhecido: {cmd}. Use !help para ver os comandos disponíveis."

    def show_help(self, args: str) -> str:
        return """
Comandos disponíveis:
!help - Mostra esta mensagem de ajuda
!execute "nome_arquivo.py arg1 arg2" - Executa um arquivo Python com argumentos opcionais
!knowledge - Mostra o conhecimento atual do sistema
"""

    def execute_python_file(self, args: str) -> str:
        try:
            parts = args.strip('"').split()
            file_name = parts[0]
            py_args = parts[1:] if len(parts) > 1 else []
            
            result = subprocess.run(["python", file_name] + py_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                return f"Execução bem-sucedida:\n{result.stdout}"
            else:
                return f"Erro na execução:\n{result.stderr}"
        except Exception as e:
            return f"Erro ao executar o arquivo: {str(e)}"

    def show_knowledge(self, args: str) -> str:
        knowledge = self.knowledge_context.knowledge_base
        if not knowledge:
            return "A base de conhecimento está vazia."
        return "Conhecimento atual:\n" + "\n".join(f"{k}: {v}" for k, v in knowledge.items())

class DialogueManager:
    def __init__(self):
        self.states = {
            "start": self.start_state,
            "greeting": self.greeting_state,
            "conversation": self.conversation_state,
            "goodbye": self.goodbye_state
        }
        self.current_state = "start"

    def start_state(self, input_data):
        # Estado inicial do diálogo
        if input_data["intent"] == "greeting":
            self.current_state = "greeting"
            return "Olá! Como posso ajudá-lo hoje?"
        else:
            return "Desculpe, não entendi. Posso ajudá-lo com algo?"

    def greeting_state(self, input_data):
        # Estado de cumprimento
        if input_data["intent"] == "question":
            self.current_state = "conversation"
            return "Ah, uma pergunta! Estou aqui para ajudar."
        elif input_data["intent"] == "goodbye":
            self.current_state = "goodbye"
            return "Tchau! Foi um prazer conversar com você."
        else:
            return "Desculpe, não entendi. Posso ajudá-lo com algo?"

    def conversation_state(self, input_data):
        # Estado de conversa
        if input_data["intent"] == "question":
            return "Ah, outra pergunta! Estou aqui para ajudar."
        elif input_data["intent"] == "goodbye":
            self.current_state = "goodbye"
            return "Tchau! Foi um prazer conversar com você."
        else:
            return "Desculpe, não entendi. Posso ajudá-lo com algo?"

    def goodbye_state(self, input_data):
        # Estado de despedida
        return "Tchau! Foi um prazer conversar com você."

    def process_input(self, input_data):
        response = self.states[self.current_state](input_data)
        return response


class SCI:
    def __init__(self, llm_client):
        self.input_layer = InputLayer()
        self.knowledge_context_layer = KnowledgeContextLayer()
        self.auto_knowledge_manager = AutoKnowledgeManager(self.knowledge_context_layer)
        self.processing_layer = ProcessingLayer(llm_client, self.auto_knowledge_manager)
        self.output_layer = OutputLayer()
        self.config_layer = ConfigLayer()
        self.command_module = CommandModule(self.knowledge_context_layer)
        self.dialogue_manager = DialogueManager()
        self.bdi_model = BDIModel(self.knowledge_context_layer)
        self.bdi_config = BDIConfig()

    def process_conversation(self, user_input):
        if user_input.startswith('!'):
            return self.command_module.process_command(user_input)
        
        
        input_data = self.input_layer.process_input(user_input)
        beliefs = Belief("intent", input_data["intent"])
        desires = Desire("respond_to_user")
        intentions = Intention("generate_response")
        self.bdi_model.add_belief(beliefs)
        self.bdi_model.add_desire(desires)
        self.bdi_model.add_intention(intentions)

        response = self.processing_layer.generate_response(input_data, self.knowledge_context_layer)
        formatted_response = self.output_layer.format_response(response)
        return formatted_response

# Exemplo de uso
if __name__ == "__main__":
    api_key = "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"
    groq_client = GroqClient(api_key)
    sci = SCI(groq_client)
    print("Bem-vindo ao SCI com gerenciamento automático de conhecimento! Digite '!help' para ver os comandos disponíveis ou comece a conversar normalmente.")
    while True:
        user_input = input("Você: ")
        if user_input.lower() == "sair":
            break
        response = sci.process_conversation(user_input)
        print(response)