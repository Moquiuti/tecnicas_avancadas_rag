# ============================================================
# CARREGAMENTO DO DOCUMENTO + CHUNKING + EMBEDDINGS + VECTOR STORE
# ============================================================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# 1. CARREGAR O DOCUMENTO
# ============================================================

loader = TextLoader("politica_cartao.txt", encoding="utf-8")
documentos = loader.load()

print(f"Quantidade de documentos carregados: {len(documentos)}")

# ============================================================
# 2. DIVIDIR EM CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documentos)

print(f"Quantidade de chunks criados: {len(chunks)}")

# ============================================================
# 3. GERAR EMBEDDINGS - CAMINHO ALTERNATIVO SEM OPENAI
# ============================================================

# Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
# Em vez de usar OpenAIEmbeddings, usamos HuggingFaceEmbeddings localmente.
# Na primeira execução, o modelo pode ser baixado automaticamente.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------
# Como seria no caminho original com OpenAI:
# ------------------------------------------------------------
# from langchain_openai import OpenAIEmbeddings
#
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )

print("Embeddings configurados com sucesso.")

# ============================================================
# 4. CRIAR BANCO VETORIAL EM MEMÓRIA
# ============================================================

# O InMemoryVectorStore armazena os vetores em memória.
# É suficiente para estudo e exemplos pequenos.

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Banco vetorial em memória criado com sucesso.")

# ============================================================
# 5. CONFIGURAR RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Retriever configurado com sucesso.")

# ============================================================
# 6. TESTAR BUSCA SEMÂNTICA
# ============================================================

pergunta = "O que acontece se uma despesa não tiver comprovante?"

documentos_recuperados = retriever.invoke(pergunta)

print("\nPergunta:")
print(pergunta)

print("\nDocumentos recuperados:")
for i, doc in enumerate(documentos_recuperados, start=1):
    print(f"\n--- Resultado {i} ---")
    print(doc.page_content)