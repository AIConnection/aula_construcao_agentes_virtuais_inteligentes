from groq import Groq

class GroqClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)

    def create_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

class CompletionConfig:
    def __init__(self):
        self.model = ""
        self.messages = []
        self.response_format={"type": "json_object"},
        self.temperature = 0.5
        self.max_tokens = 1024
        self.top_p = 0.65
        self.stream = True
        self.stop = None

    def to_dict(self):
        return {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": False,
            "stop": None,
        }

def generate(text:str):
    api_key='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'
    config = CompletionConfig()
    config.model = "llama3-groq-70b-8192-tool-use-preview"
    config.messages = [
        {
            "role": "user",
            "content": text
        },
        {
          "role": "system",
          "content":"você como especialista em marketing e comportamento deve avaliar através de reclamações se existe probabilidade do consumidor voltar a fazer negócios. Avalie se é valida a reclamação e sugira uma ação rentavel."
        },
        {
            "role": "assistant",
            "content": 
                """
                extraia as entidades, desejos e intenções, faça uma analise de sentimento (POSITIVO, NEGATIVO, NEUTRO) e classifique o texto com rótulos semanticos..                
                
                entidades -> lista de string com as entidades extraidas. (deve possuir pelo menos 2 entidades). (lista). (obrigatório).
                entidades.entidade-> termo ou expressão que representa um objeto no mundo real. (objeto). (obrigatório).
                entidades.entidade.descricao-> descrição do termo ou expressão extraido. (texto). (obrigatório).
                entidades.entidade.probabilidade-> probabilidade do termo ou expressão extraido representar o objeto no mundo real. (ponto flutuante min 0, max 1). (obrigatório).
                
                desejos -> lista de string com os desejos extraidos. (deve possuir pelo menos 1 desejo). (lista). (obrigatório).
                desejos.desejo-> estado ou condição de uma entidade, representa algo que se gostaria de fazer ou alcançar, os desejos podem ser implicitos ou explicitos. (objeto) .(obrigatório).
                desejos.desejo.descricao-> descricao do desejo extraido. (texto). (obrigatório).
                desejos.desejo.tipo-> categoria: IMPLICITO, EXPLICITO. (categoria). (obrigatório).
                desejos.desejo.probabilidade-> probabilidade do desejo extraido. (ponto flutuante min 0, max 1). (obrigatório).
                
                sentiment_analisys -> tom emocional ou sentimental extraido do texto (objeto). (obrigatório).
                sentiment_analisys.description -> categoria: POSITIVO, NEGATIVO, NEUTRO. (objeto). (obrigatório).
                sentiment_analisys.probabilidade -> probabilidade do desejo sentimento extraido. (ponto flutuante min 0, max 1). (obrigatório).
                
                intencao -> a intenção é o objetivo ou propósito por trás de uma mensagem ou texto. 
                    É a razão pela qual alguém está escrevendo ou falando, e é frequentemente expressa por meio de palavras-chave, frases ou sentenças que indicam o que alguém está procurando ou pretendendo. (objeto). (obrigatório).
                intencao.descricao-> descricao do objetivo ou propósito. (texto). (obrigatório).
                
                rotulos -> caracteristicas do conteúdo. (deve possuir pelo menos 2 categorias). (texto), (obrigatório).
                
                analise_semantica-> os significados das entidades, desejos, sentimento e intenções do conteúdo. (texto). (obrigatório).
                
                analise_comportamento -> resultado da analise do especialista. (texto). (obrigatório).
                
                acao_imediata-> o que deve ser feito nesse cenario. (texto). (obrigatório).
                
                instrução 1: gere os valores em portugês do brasil com traduções apropriadas semanticamente.
                instrução 2: gere um json formatado.
                
                {
                    "entidades": [
                        "entidade": {
                            "descricao":"string",
                            "probabilidade":"number (0-1)"
                        },
                        "entidade": {
                            "descricao":"string",
                            "probabilidade":"number (0-1)"
                        }
                    ],
                    "desejos": [
                        "desejo": {
                            "descricao":"string",
                            "tipo":"string (implicito, explicito)",
                            "probabilidade":"number (0-1)"
                        },
                        "desejo": {
                            "descricao":"string",
                            "tipo":"string (implicito, explicito)",
                            "probabilidade":"number (0-1)"
                        }
                    ],
                    "analise_sentimento": {
                        "descricao": "string (positivo, negativo, neutro)",
                        "probabilidade": "number (0-1)"
                    }
                    "intencao": {
                        "descricao": "string"
                    }
                    "rotulos": [
                        "string",
                        "string",
                        "string"
                    ],
                    "analise_semantica": "string",
                    "analise_comportamento": "string",
                    "acao_imediata":""
                }
                """
        }
    ]

    client = GroqClient(api_key)
    
    completion = client.create_completion(**config.to_dict())
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    response = generate(
        """
Eu estou muito desapontado com a minha experiência com a empresa XYZ. Eu comprei um produto deles há alguns meses e desde então, tenho tido problemas constantes com a qualidade do produto. A última vez que o usei, ele quebrou e agora estou sem o produto que eu precisava. Eu esperava que a empresa fosse mais responsável e garantisse a qualidade dos seus produtos. Eu sinto que eu estou perdendo tempo e dinheiro com essa empresa.
Eu gostaria que a empresa resolvesse esse problema e me devolvesse o dinheiro que eu gastei com o produto. Eu também gostaria que eles melhorassem a qualidade dos seus produtos para que os clientes não tenham que passar por isso novamente.
Eu estou muito frustrado com essa situação e espero que a empresa tome medidas para resolver esse problema. Eu gostaria de ter uma resposta rápida e eficaz para resolver essa situação.
        """)
    print(response)
    
    
