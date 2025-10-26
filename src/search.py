import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
    """
    Inicializa a cadeia de busca vetorial e geração de resposta.
    
    Returns:
        dict: Dicionário com as funções de busca e geração de resposta, ou None se houver erro
    """
    try:
        required_vars = ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME")
        for var in required_vars:
            if not os.getenv(var):
                print(f"Erro: Variável de ambiente {var} não está definida")
                return None

        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )

        store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            temperature=0
        )

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        def search_and_answer(question):
            """
            Busca documentos relevantes e gera resposta baseada no contexto.
            
            Args:
                question (str): Pergunta do usuário
                
            Returns:
                str: Resposta gerada pela LLM
            """
            try:
                results = store.similarity_search_with_score(question, k=10)
                
                if not results:
                    return "Não tenho informações necessárias para responder sua pergunta."
                
                contexto = "\n\n".join([doc.page_content for doc, score in results])
                
                chain = prompt_template | llm
                response = chain.invoke({
                    "contexto": contexto,
                    "pergunta": question
                })
                
                return response.content
                
            except Exception as e:
                print(f"Erro durante a busca: {e}")
                return "Ocorreu um erro durante a busca. Tente novamente."

        return {
            "search_and_answer": search_and_answer,
            "store": store,
            "llm": llm
        }

    except Exception as e:
        print(f"Erro na inicialização: {e}")
        return None