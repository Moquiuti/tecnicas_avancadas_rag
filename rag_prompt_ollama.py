# ============================================================
# RAG COM PROMPT TEMPLATE + CHATOLLAMA
# ============================================================

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# ============================================================
# 0. CONFIGURAÇÃO INICIAL
# ============================================================

load_dotenv()

# Observação:
# O curso utiliza OpenAI e LangSmith.
# Neste ambiente, a chamada para OpenAI retornou erro 429 - insufficient_quota,
# e o LangSmith retornou erro 403 - Forbidden.
# Por isso, esta implementação foi adaptada para usar:
# - HuggingFaceEmbeddings no lugar de OpenAIEmbeddings
# - ChatOllama no lugar de ChatOpenAI
# - LangSmith desligado no arquivo .env

# ------------------------------------------------------------
# Como seria no caminho original com OpenAI e LangSmith:
# ------------------------------------------------------------
# .env:
# OPENAI_API_KEY=sua_chave_openai
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=sua_chave_langsmith
# LANGSMITH_PROJECT=langchain-rag-avancado
#
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ============================================================
# 1. CARREGAR DOCUMENTO
# ============================================================

loader = TextLoader("politica_cartao.txt", encoding="utf-8")
documentos = loader.load()

print(f"Documentos carregados: {len(documentos)}")


# ============================================================
# 2. FAZER CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documentos)

print(f"Chunks criados: {len(chunks)}")


# ============================================================
# 3. GERAR EMBEDDINGS - CAMINHO ALTERNATIVO SEM OPENAI
# ============================================================

# Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
# Em vez de usar OpenAIEmbeddings, usamos HuggingFaceEmbeddings localmente.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings configurados com HuggingFaceEmbeddings.")


# ============================================================
# 4. CRIAR BANCO VETORIAL INMEMORY
# ============================================================

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Banco vetorial InMemory e retriever configurados.")


# ============================================================
# 5. FORMATAR DOCUMENTOS RECUPERADOS
# ============================================================

def formatar_documentos(documentos_recuperados):
    return "\n\n".join(doc.page_content for doc in documentos_recuperados)


# ============================================================
# 6. MONTAR PROMPT TEMPLATE
# ============================================================

# Aqui montamos o prompt com duas mensagens:
# - system: define o comportamento do assistente
# - human: envia o contexto recuperado e a pergunta do usuário

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente especializado em responder perguntas com base em documentos. "
        "Responda somente com base no contexto fornecido. "
        "Se a resposta não estiver no contexto, diga: 'Não encontrei essa informação no documento.' "
        "Seja direto, claro e fiel ao conteúdo recuperado."
    ),
    (
        "human",
        "Contexto recuperado:\n{context}\n\nPergunta do usuário:\n{question}"
    )
])

print("Prompt template configurado.")


# ============================================================
# 7. CONFIGURAR MODELO - CAMINHO ALTERNATIVO COM OLLAMA
# ============================================================

# Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
# Em vez de usar ChatOpenAI, usamos ChatOllama rodando localmente.

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

print("Modelo ChatOllama configurado.")


# ============================================================
# 8. MONTAR CADEIA RAG COM LCEL
# ============================================================

chain = (
        {
            "context": retriever | formatar_documentos,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)

print("Cadeia RAG configurada.")


# ============================================================
# 9. EXECUTAR TESTE
# ============================================================

perguntas = [
    "Para que o cartão de crédito corporativo pode ser utilizado?",
    "O cartão pode ser usado para compras pessoais?",
    "O que acontece com despesas sem comprovante?",
    "Qual é o limite mensal do cartão?",
    "O documento fala sobre reembolso de combustível?"
]

for pergunta in perguntas:
    print("\n" + "=" * 80)
    print(f"Pergunta: {pergunta}")
    print("-" * 80)

    resposta = chain.invoke(pergunta)

    print("Resposta:")
    print(resposta)