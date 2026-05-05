# -*- coding: utf-8 -*-
"""
Pipeline RAG Local Completo - PDFs + Ollama + FAISS + LangChain

Implementa técnicas avançadas:
- Query Rewriting
- MultiQuery Retriever
- HyDE (Hypothetical Document Embeddings)

Autor: Adaptado do notebook Jupyter
"""

import os
import warnings
from pathlib import Path

# Configurações iniciais
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['USER_AGENT'] = 'RAGProject/1.0'

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PDF_DIR = Path("./pdfs")
VECTOR_DB_DIR = Path("./vectorstore_faiss")

# Modelos locais via Ollama
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2:3b"  # Corrigido de gemma3:4b

PDF_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🚀 PIPELINE RAG LOCAL COMPLETO")
print("=" * 80)
print(f"📁 Diretório dos PDFs: {PDF_DIR.resolve()}")
print(f"💾 Diretório do VectorStore: {VECTOR_DB_DIR.resolve()}")
print(f"🎯 Modelo Embeddings: {EMBEDDING_MODEL}")
print(f"🤖 Modelo LLM: {LLM_MODEL}")

# ============================================================
# 1. CARREGAR PDFs
# ============================================================

print("\n" + "=" * 80)
print("📄 ETAPA 1: CARREGANDO PDFs")
print("=" * 80)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=str(PDF_DIR),
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True,
)

documents = loader.load()

print(f"✅ Páginas/documentos carregados: {len(documents)}")

if not documents:
    print("❌ Nenhum PDF encontrado. Adicione arquivos na pasta ./pdfs")
    exit(1)

print(f"\n📊 Exemplo de metadados do primeiro documento:")
print(f"   Fonte: {documents[0].metadata.get('source')}")
print(f"   Página: {documents[0].metadata.get('page')}")
print(f"\n📝 Prévia do conteúdo:")
print(f"   {documents[0].page_content[:300]}...")

# ============================================================
# 2. CHUNKING COM TOKENIZER HUGGING FACE
# ============================================================

print("\n" + "=" * 80)
print("✂️ ETAPA 2: CHUNKING COM TOKENIZER HUGGING FACE")
print("=" * 80)

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

TOKENIZER_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

print(f"⚙️ Carregando tokenizer: {TOKENIZER_NAME}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Total de documentos originais: {len(documents)}")
print(f"✅ Total de chunks gerados: {len(chunks)}")
print(f"\n📊 Análise do primeiro chunk:")
print(f"   Tamanho: {len(chunks[0].page_content)} caracteres")
print(f"   Prévia: {chunks[0].page_content[:300]}...")

# ============================================================
# 3. EMBEDDINGS LOCAIS COM BGE-M3 VIA OLLAMA
# ============================================================

print("\n" + "=" * 80)
print("🧠 ETAPA 3: CONFIGURANDO EMBEDDINGS LOCAIS")
print("=" * 80)

from langchain_ollama import OllamaEmbeddings

print(f"⚙️ Inicializando embeddings: {EMBEDDING_MODEL}")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Teste rápido
print("🧪 Testando geração de embeddings...")
teste_vetor = embeddings.embed_query("Teste de embeddings com BGE-M3 via Ollama.")
print(f"✅ Dimensão do vetor: {len(teste_vetor)}")
print(f"   Amostra: {teste_vetor[:5]}")

# ============================================================
# 4. CRIAR E SALVAR VECTORSTORE COM FAISS
# ============================================================

print("\n" + "=" * 80)
print("💾 ETAPA 4: CRIANDO VECTORSTORE COM FAISS")
print("=" * 80)

from langchain_community.vectorstores import FAISS

print("⚙️ Gerando embeddings para todos os chunks...")
print("   (isso pode demorar alguns minutos dependendo da quantidade)")

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

print(f"✅ VectorStore criado com {len(chunks)} chunks")

print("💾 Salvando em disco...")
vectorstore.save_local(str(VECTOR_DB_DIR))

print(f"✅ VectorStore salvo em: {VECTOR_DB_DIR.resolve()}")

# ============================================================
# 5. CARREGAR VECTORSTORE E CRIAR RETRIEVER
# ============================================================

print("\n" + "=" * 80)
print("🔄 ETAPA 5: CARREGANDO VECTORSTORE E CRIANDO RETRIEVER")
print("=" * 80)

vectorstore = FAISS.load_local(
    folder_path=str(VECTOR_DB_DIR),
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

print("✅ Retriever criado com sucesso (top-k=4)")

# ============================================================
# 6. TESTAR RETRIEVER
# ============================================================

print("\n" + "=" * 80)
print("🧪 ETAPA 6: TESTANDO RETRIEVER")
print("=" * 80)

consulta = "Qual é o tema principal dos documentos?"

print(f"❓ Consulta: {consulta}")

docs_retornados = retriever.invoke(consulta)

print(f"✅ Documentos retornados: {len(docs_retornados)}")

for i, doc in enumerate(docs_retornados, start=1):
    print(f"\n📄 Documento {i}:")
    print(f"   Fonte: {doc.metadata.get('source')}")
    print(f"   Página: {doc.metadata.get('page')}")
    print(f"   Conteúdo: {doc.page_content[:200]}...")

# ============================================================
# 7. CONFIGURAR LLM LOCAL
# ============================================================

print("\n" + "=" * 80)
print("🤖 ETAPA 7: CONFIGURANDO LLM LOCAL")
print("=" * 80)

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.2,
)

print(f"⚙️ LLM configurado: {LLM_MODEL}")

print("🧪 Testando LLM...")
resposta_teste = llm.invoke("Responda em uma frase: o que é RAG?")
print(f"✅ Resposta: {resposta_teste.content}")

# ============================================================
# 8. CRIAR CADEIA RAG BÁSICA
# ============================================================

print("\n" + "=" * 80)
print("⛓️ ETAPA 8: CRIANDO CADEIA RAG BÁSICA")
print("=" * 80)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(
        f"[Fonte: {Path(doc.metadata.get('source', 'unknown')).name} | Página: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )

rag_prompt = ChatPromptTemplate.from_template(
    """Você é um assistente técnico. Responda com base apenas no contexto fornecido.

Contexto:
{context}

Pergunta:
{question}

Resposta em português, objetiva e bem estruturada:
"""
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("✅ Cadeia RAG básica criada")

print(f"\n💬 Executando RAG com consulta: {consulta}")
resposta = rag_chain.invoke(consulta)

print("\n📝 RESPOSTA:")
print("=" * 80)
print(resposta)
print("=" * 80)

# ============================================================
# 9. QUERY REWRITING
# ============================================================

print("\n" + "=" * 80)
print("🔄 ETAPA 9: QUERY REWRITING")
print("=" * 80)

query_rewriter_llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)

rewrite_prompt = ChatPromptTemplate.from_template(
    """Reescreva a pergunta abaixo para melhorar a busca semântica em documentos PDF.

Regras:
- mantenha o sentido original;
- não responda à pergunta;
- gere apenas uma versão refinada da consulta;
- remova ambiguidades;
- preserve termos técnicos importantes.

Pergunta original:
{question}

Consulta refinada:
"""
)

query_rewriter_chain = rewrite_prompt | query_rewriter_llm | StrOutputParser()

print(f"❓ Consulta original: {consulta}")

consulta_refinada = query_rewriter_chain.invoke({"question": consulta})

print(f"✅ Consulta refinada: {consulta_refinada}")

# Comparar recuperação
docs_original = retriever.invoke(consulta)
docs_rewritten = retriever.invoke(consulta_refinada)

print(f"\n📊 Comparação:")
print(f"   Docs (original): {len(docs_original)}")
print(f"   Docs (refinada): {len(docs_rewritten)}")

print("\n💬 Executando RAG com consulta refinada...")
resposta_rewritten = rag_chain.invoke(consulta_refinada)

print("\n📝 RESPOSTA COM QUERY REWRITING:")
print("=" * 80)
print(resposta_rewritten)
print("=" * 80)

# ============================================================
# 10. MULTIQUERY RETRIEVER
# ============================================================

print("\n" + "=" * 80)
print("🔍 ETAPA 10: MULTIQUERY RETRIEVER")
print("=" * 80)

from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm,
)

print("⚙️ MultiQuery Retriever configurado")

print(f"🔍 Buscando com MultiQuery: {consulta}")

docs_multiquery = multiquery_retriever.invoke(consulta)

print(f"✅ Documentos consolidados: {len(docs_multiquery)}")

# RAG com MultiQuery
rag_chain_multiquery = (
    {
        "context": multiquery_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("\n💬 Executando RAG com MultiQuery...")
resposta_multiquery = rag_chain_multiquery.invoke(consulta)

print("\n📝 RESPOSTA COM MULTIQUERY:")
print("=" * 80)
print(resposta_multiquery)
print("=" * 80)

# ============================================================
# 11. HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS)
# ============================================================

print("\n" + "=" * 80)
print("🎭 ETAPA 11: HyDE (HYPOTHETICAL DOCUMENT EMBEDDINGS)")
print("=" * 80)

hyde_prompt = ChatPromptTemplate.from_template(
    """Gere um parágrafo hipotético, técnico e objetivo, que poderia aparecer em um documento PDF respondendo à pergunta abaixo.

Não diga que é hipotético.
Não use introduções.
Escreva apenas o parágrafo.

Pergunta:
{question}

Parágrafo:
"""
)

hyde_chain = hyde_prompt | llm | StrOutputParser()

print("⚙️ Gerando documento hipotético...")
hyde_document = hyde_chain.invoke({"question": consulta})

print(f"\n📝 Documento hipotético gerado:")
print(f"   {hyde_document[:300]}...")

print(f"\n🔍 Buscando documentos com o texto hipotético...")
docs_hyde = retriever.invoke(hyde_document)

print(f"✅ Documentos retornados via HyDE: {len(docs_hyde)}")

# RAG com HyDE
def hyde_retrieve(question: str):
    hypothetical_doc = hyde_chain.invoke({"question": question})
    return retriever.invoke(hypothetical_doc)

rag_chain_hyde = (
    {
        "context": hyde_retrieve | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("\n💬 Executando RAG com HyDE...")
resposta_hyde = rag_chain_hyde.invoke(consulta)

print("\n📝 RESPOSTA COM HYDE:")
print("=" * 80)
print(resposta_hyde)
print("=" * 80)

# ============================================================
# 12. COMPARAÇÃO FINAL
# ============================================================

print("\n" + "=" * 80)
print("📊 COMPARAÇÃO FINAL DAS TÉCNICAS")
print("=" * 80)

print("\n1️⃣ CONSULTA ORIGINAL:")
print("-" * 80)
print(resposta)

print("\n2️⃣ QUERY REWRITING:")
print("-" * 80)
print(resposta_rewritten)

print("\n3️⃣ MULTIQUERY:")
print("-" * 80)
print(resposta_multiquery)

print("\n4️⃣ HYDE:")
print("-" * 80)
print(resposta_hyde)

# ============================================================
# CONCLUSÃO
# ============================================================

print("\n" + "=" * 80)
print("✅ PIPELINE RAG COMPLETO EXECUTADO COM SUCESSO!")
print("=" * 80)

print(f"""
📊 Estatísticas:
   • PDFs processados: {len(set(doc.metadata.get('source') for doc in documents))}
   • Páginas totais: {len(documents)}
   • Chunks gerados: {len(chunks)}
   • Dimensão dos embeddings: {len(teste_vetor)}
   • Modelo LLM: {LLM_MODEL}
   • Modelo Embeddings: {EMBEDDING_MODEL}

🎯 Técnicas implementadas:
   ✅ RAG Básico
   ✅ Query Rewriting
   ✅ MultiQuery Retriever
   ✅ HyDE (Hypothetical Document Embeddings)

💾 Arquivos salvos:
   • VectorStore FAISS: {VECTOR_DB_DIR.resolve()}
   
💡 Próximos passos:
   • Testar com perguntas diferentes
   • Adicionar mais PDFs
   • Implementar reranking
   • Criar interface de chat
""")

print("🎉 Obrigado por usar o Pipeline RAG!")

