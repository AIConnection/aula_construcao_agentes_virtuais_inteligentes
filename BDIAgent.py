import json
from abc import ABC, abstractmethod
from groq import Groq

class CompletionConfig:
    def __init__(self):
        self.api_key='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'
        self.model = "llama3-groq-70b-8192-tool-use-preview"
        self.messages = []
        self.tools = []
        self.tool_choice= "auto"
        self.temperature = 0
        self.max_tokens = 4096
        self.top_p = 0.65
        self.stream = False
        self.seed = 5465

class LLM():
    def __init__(self, config:CompletionConfig):
        self.config = config
        self.client = Groq(api_key=config.api_key)
    
    def complete(self, text:str):
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[s.replace("[TEXT]", text) for s in self.config.messages],
            tools=self.config.tools,
            tool_choice=self.config.tool_choice,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content

class Belief(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
        
        config = CompletionConfig()
        config.messages = [
            {
                "role": "user",
                "content":
                    """
                    [CONTEXT]
                    """
            },
            {
            "role": "system",
            "content":
                """
                avalie se é verdadeiro ou falso
                retone um booleano somente, não faça comentários.
                """
            },
            {
                "role": "assistant",
                "content": 
                    """
                        return boolean(True|False)
                    """
            }
        ]
        self.llm = LLM(config=config)

    def is_true(self, context):

        return eval(self.llm.complete(context))


class Desire(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
        config = CompletionConfig()
        config.messages = [
            {
                "role": "user",
                "content":
                    """
                    [CONTEXT]
                    """
            },
            {
            "role": "system",
            "content":
                """
                avalie a probabilidade.
                retone um float somente, não faça comentários.
                """
            },
            {
                "role": "assistant",
                "content": 
                    """
                        return number(min 0, max 1)
                    """
            }
        ]
        
        self.llm = LLM(config=config)

    def is_satisfied(self, context):
        self.llm = self.llm.complete(context)


# Interface para as intenções
class Intention(ABC):
    @abstractmethod
    def evaluate_preconditions(self):
        pass

    @abstractmethod
    def evaluate_goals(self):
        pass

    def is_applicable(self):
        return self.evaluate_preconditions() and self.evaluate_goals()

    @abstractmethod
    def perform(self):
        pass


# Interface para o agente
class Agent(ABC):
    def __init__(self, acoes):
        self.acoes = acoes
        self.beliefs = []
        self.desires = []
        self.intentions = []

    @abstractmethod
    def add_belief(self, belief):
        pass

    @abstractmethod
    def add_desire(self, desire):
        pass

    @abstractmethod
    def add_intention(self, intention):
        pass

    @abstractmethod
    def execute_intentions(self):
        pass

    def load_config(self, filename):
        with open(filename) as f:
            config = json.load(f)

        for belief_config in config['agent']['beliefs']:
            belief = self.create_belief(belief_config)
            self.add_belief(belief)

        for desire_config in config['agent']['desires']:
            desire = self.create_desire(desire_config)
            self.add_desire(desire)

        for intention_config in config['agent']['intentions']:
            conditions = [self.beliefs[i] for i, belief in enumerate(config['agent']['beliefs']) if belief['name'] in intention_config['conditions']]
            goals = [self.desires[i] for i, desire in enumerate(config['agent']['desires']) if desire['name'] in intention_config['goals']]
            action = self.acoes.get(intention_config['action'])
            if action is None:
                raise ValueError(f"Ação desconhecida: {intention_config['action']}")
            intention = self.create_intention(intention_config['name'], action, conditions, goals)
            self.add_intention(intention)

    @abstractmethod
    def create_belief(self, config):
        pass
    
    @abstractmethod
    def create_desire(self, config):
        pass

    @abstractmethod
    def create_intention(self, name, action, conditions, goals):
        pass

    def load_config_from_file(self, filename):
        try:
            self.load_config(filename)
        except FileNotFoundError:
            print(f"Arquivo {filename} não encontrado")
        except json.JSONDecodeError:
            print(f"Erro ao ler arquivo {filename}")


# Implementação de uma crença simples
class SimpleBelief(Belief):
    def __init__(self, name, description, value):
        self.name = name
        self.description = description
        self.value = value

    def is_true(self):
        return True


# Implementação de um Desejo simples
class SimpleDesire(Desire):
    def __init__(self, name, description, value):
        self.name = name
        self.description = description
        self.value = value

    def is_satisfied(self):
       return 0.5


# Implementação de uma intenção simples
class SimpleIntention(Intention):
    def __init__(self, name, action, conditions=None, goals=None):
        self.name = name
        self.action = action
        self.conditions = conditions if conditions else []
        self.goals = goals if goals else []

    def evaluate_preconditions(self):
        for condition in self.conditions:
            if not condition.is_true():
                return False
        return True

    def evaluate_goals(self):
        for goal in self.goals:
            if not goal.is_satisfied():
                return False
        return True

    def perform(self):
        if self.is_applicable():
            self.action()

# Implementação de um agente simples
class SimpleAgent(Agent):
    
    def __init__(self, acoes, config_file_name):
        super().__init__(acoes)
        self.config_file_name = config_file_name
    
    def load(self):
        self.load_config_from_file(self.config_file_name)
    
    def create_belief(self, config):
        return SimpleBelief(config['name'], config['description'], 1)

    def create_desire(self, config):
        return SimpleDesire(config['name'], config['description'])

    def create_intention(self, name, action, conditions, goals):
        return SimpleIntention(name, action, conditions, goals)
    
    def add_belief(self, belief):
        self.beliefs.append(belief)

    def add_desire(self, desire):
        self.desires.append(desire)

    def add_intention(self, intention):
        self.intentions.append(intention)

    def execute_intentions(self):
        for intention in self.intentions:
            intention.perform()
