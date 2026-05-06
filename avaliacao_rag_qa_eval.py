# ============================================================
# ATIVIDADE - AVALIAÇÃO AUTOMATIZADA DE PIPELINE RAG
# QA-EVAL + GERAÇÃO AUTOMÁTICA DE PERGUNTAS E RESPOSTAS
# ============================================================

# Objetivo:
# - Preparar um dataset de avaliação com query/answer
# - Gerar respostas sem RAG
# - Gerar respostas com RAG
# - Avaliar automaticamente as respostas usando QAEvalChain
# - Gerar pares automáticos de pergunta/resposta a partir dos chunks
# - Comparar resultados e analisar o desempenho
#
# Observação importante:
# O curso utiliza OpenAI, QAEvalChain e QAGenerationChain/QAGenerateChain.
# Neste ambiente, a OpenAI retornou erro 429 - insufficient_quota.
# Por isso, mantemos a arquitetura do curso, mas usamos ChatOllama local.
#
# Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
# A avaliação com LLM local pode ser menos precisa do que com modelos maiores,
# mas preserva a proposta técnica sem custo de API.
# ============================================================


import os
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# 0. CONFIGURAÇÕES GERAIS
# ============================================================

os.environ["LANGSMITH_TRACING"] = "false"

BASE_DIR = Path("C:/ia/rag")
PDF_DIR = BASE_DIR / "pdfs"
DADOS_DIR = BASE_DIR / "dados"
VECTORSTORE_DIR = BASE_DIR / "vectorstore_faiss_eval"

MODELO_LLM = "llama3.2:3b"
MODELO_EMBEDDINGS = "bge-m3"

print("=" * 80)
print("🚀 AVALIAÇÃO AUTOMATIZADA DE PIPELINE RAG")
print("=" * 80)
print(f"🤖 Modelo LLM local: {MODELO_LLM}")
print(f"🧠 Modelo de embeddings local: {MODELO_EMBEDDINGS}")
print(f"📁 Diretório PDFs: {PDF_DIR}")
print(f"💾 VectorStore: {VECTORSTORE_DIR}")


# ============================================================
# 1. CONFIGURAR LLM E EMBEDDINGS LOCAIS
# ============================================================

# Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
# No caminho original do curso, seria usado ChatOpenAI e OpenAIEmbeddings.
#
# Exemplo original com OpenAI:
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOllama(
    model=MODELO_LLM,
    temperature=0
)

embeddings = OllamaEmbeddings(
    model=MODELO_EMBEDDINGS
)

print("\n✅ LLM e embeddings locais configurados.")


# ============================================================
# 2. CARREGAR DOCUMENTOS
# ============================================================

print("\n" + "=" * 80)
print("📄 ETAPA 1: CARREGANDO DOCUMENTOS")
print("=" * 80)

documentos = []

# Carrega PDFs, se existirem
if PDF_DIR.exists():
    pdfs = list(PDF_DIR.glob("*.pdf"))

    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            docs_pdf = loader.load()
            documentos.extend(docs_pdf)
            print(f"✅ PDF carregado: {pdf.name} | páginas: {len(docs_pdf)}")
        except Exception as erro:
            print(f"⚠️ Falha ao carregar PDF {pdf.name}: {erro}")

# Carrega TXTs da pasta dados, se existirem
if DADOS_DIR.exists():
    try:
        txt_loader = DirectoryLoader(
            path=str(DADOS_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs_txt = txt_loader.load()
        documentos.extend(docs_txt)
        print(f"✅ TXTs carregados: {len(docs_txt)}")
    except Exception as erro:
        print(f"⚠️ Falha ao carregar TXTs: {erro}")

if not documentos:
    raise RuntimeError("Nenhum documento foi carregado. Verifique as pastas pdfs/ e dados/.")

print(f"\n📊 Total de documentos carregados: {len(documentos)}")
print("📝 Prévia do primeiro documento:")
print(documentos[0].page_content[:500])


# ============================================================
# 3. CHUNKING
# ============================================================

print("\n" + "=" * 80)
print("✂️ ETAPA 2: REALIZANDO CHUNKING")
print("=" * 80)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documentos)

print(f"✅ Total de chunks gerados: {len(chunks)}")
print("📝 Prévia do primeiro chunk:")
print(chunks[0].page_content[:500])


# ============================================================
# 4. CRIAR OU CARREGAR VECTORSTORE FAISS
# ============================================================

print("\n" + "=" * 80)
print("💾 ETAPA 3: CRIANDO/CARREGANDO VECTORSTORE FAISS")
print("=" * 80)

if VECTORSTORE_DIR.exists():
    print("📦 VectorStore existente encontrado. Carregando do disco...")

    vectorstore = FAISS.load_local(
        folder_path=str(VECTORSTORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

else:
    print("⚙️ Criando novo VectorStore FAISS...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(str(VECTORSTORE_DIR))
    print("✅ VectorStore salvo em disco.")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("✅ Retriever configurado com k=3.")


# ============================================================
# 5. DATASET MANUAL DE AVALIAÇÃO
# ============================================================

print("\n" + "=" * 80)
print("🧪 ETAPA 4: PREPARANDO DATASET MANUAL DE AVALIAÇÃO")
print("=" * 80)

# O dataset precisa seguir o formato esperado:
# query  = pergunta
# answer = resposta esperada/gabarito

evaluation_dataset = [
    {
        "query": "O cartão corporativo pode ser usado para compras pessoais?",
        "answer": "Não. O cartão corporativo deve ser utilizado apenas para despesas relacionadas ao trabalho e não para compras pessoais."
    },
    {
        "query": "O que acontece com despesas sem comprovante?",
        "answer": "Despesas sem comprovante podem ser recusadas e cobradas do colaborador."
    },
    {
        "query": "Viagens corporativas precisam de aprovação?",
        "answer": "Sim. Viagens corporativas devem ser previamente aprovadas pelo gestor."
    },
    {
        "query": "O que é necessário no processo de compras corporativas?",
        "answer": "O processo deve conter justificativa, fornecedor e comprovante."
    },
    {
        "query": "Fornecedores precisam estar cadastrados antes da contratação?",
        "answer": "Sim. Fornecedores devem estar cadastrados antes da contratação."
    }
]

print(f"✅ Dataset manual criado com {len(evaluation_dataset)} perguntas.")
for item in evaluation_dataset:
    print(f"- {item['query']}")


# ============================================================
# 6. FUNÇÃO PARA FORMATAR DOCUMENTOS RECUPERADOS
# ============================================================

def formatar_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# 7. CHAIN SEM RAG
# ============================================================

print("\n" + "=" * 80)
print("🤖 ETAPA 5: CONFIGURANDO GERAÇÃO SEM RAG")
print("=" * 80)

prompt_sem_rag = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente de perguntas e respostas. "
        "Responda da melhor forma possível, mas sem acesso a documentos externos."
    ),
    (
        "human",
        "{question}"
    )
])

chain_sem_rag = (
        {"question": RunnablePassthrough()}
        | prompt_sem_rag
        | llm
        | StrOutputParser()
)

def generate_without_rag(question: str) -> str:
    return chain_sem_rag.invoke(question)

print("✅ Chain sem RAG configurada.")


# ============================================================
# 8. CHAIN COM RAG
# ============================================================

print("\n" + "=" * 80)
print("📚 ETAPA 6: CONFIGURANDO GERAÇÃO COM RAG")
print("=" * 80)

prompt_com_rag = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente especializado em responder perguntas com base em documentos. "
        "Use somente o contexto fornecido. "
        "Se a resposta não estiver no contexto, diga que não encontrou a informação nos documentos."
    ),
    (
        "human",
        "Contexto recuperado:\n{context}\n\nPergunta:\n{question}"
    )
])

chain_com_rag = (
        {
            "context": retriever | formatar_documentos,
            "question": RunnablePassthrough()
        }
        | prompt_com_rag
        | llm
        | StrOutputParser()
)

def generate_with_rag(question: str) -> str:
    return chain_com_rag.invoke(question)

print("✅ Chain com RAG configurada.")


# ============================================================
# 9. IMPORTAR QA-EVAL E QA-GENERATION COM FALLBACK
# ============================================================

print("\n" + "=" * 80)
print("📏 ETAPA 7: CONFIGURANDO QA-EVAL")
print("=" * 80)

# Dependendo da versão do LangChain, essas classes podem estar em módulos diferentes.
# Na versão atual, parte do ecossistema clássico fica em langchain_classic.

try:
    from langchain_classic.evaluation.qa import QAEvalChain
    from langchain_classic.evaluation.qa.generate_chain import QAGenerateChain

    print("✅ Imports realizados via langchain_classic.")

except Exception:
    try:
        from langchain.evaluation.qa import QAEvalChain, QAGenerateChain

        print("✅ Imports realizados via langchain.evaluation.qa.")

    except Exception as erro:
        raise ImportError(
            "Não foi possível importar QAEvalChain/QAGenerateChain. "
            "Verifique a instalação do langchain-classic."
        ) from erro


eval_chain = QAEvalChain.from_llm(llm)

print("✅ QAEvalChain configurado.")


# ============================================================
# 10. FUNÇÃO DE AVALIAÇÃO ITERATIVA
# ============================================================

def evaluate_pipeline(dataset, generator_function, nome_pipeline: str):
    print("\n" + "=" * 80)
    print(f"🔎 AVALIANDO PIPELINE: {nome_pipeline}")
    print("=" * 80)

    predictions = []
    references = []

    for item in dataset:
        query = item["query"]
        expected_answer = item["answer"]

        print(f"\n❓ Pergunta: {query}")

        generated_answer = generator_function(query)

        print("🤖 Resposta gerada:")
        print(generated_answer)

        predictions.append({
            "query": query,
            "result": generated_answer
        })

        references.append({
            "query": query,
            "answer": expected_answer
        })

    print("\n📏 Comparando respostas geradas com gabaritos usando QAEvalChain...")

    graded_outputs = eval_chain.evaluate(
        references,
        predictions
    )

    return graded_outputs, predictions, references


# ============================================================
# 11. CÁLCULO DE ACURÁCIA
# ============================================================

def compute_accuracy(results):
    if not results:
        return 0

    correct = 0

    for item in results:
        text = str(item).upper()

        if "CORRECT" in text and "INCORRECT" not in text:
            correct += 1

    return correct / len(results)


def print_results(results, titulo):
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)

    for i, result in enumerate(results, start=1):
        print(f"\nResultado {i}:")
        print(result)

    print(f"\n✅ Acurácia estimada: {compute_accuracy(results):.2%}")


# ============================================================
# 12. EXECUTAR AVALIAÇÃO SEM RAG E COM RAG
# ============================================================

results_no_rag, predictions_no_rag, refs_no_rag = evaluate_pipeline(
    evaluation_dataset,
    generate_without_rag,
    "SEM RAG"
)

results_rag, predictions_rag, refs_rag = evaluate_pipeline(
    evaluation_dataset,
    generate_with_rag,
    "COM RAG"
)

print_results(results_no_rag, "📊 RESULTADOS - SEM RAG")
print_results(results_rag, "📊 RESULTADOS - COM RAG")


# ============================================================
# 13. GERAÇÃO AUTOMÁTICA DE PARES QA COM QAGenerateChain
# ============================================================

print("\n" + "=" * 80)
print("🧬 ETAPA 8: GERANDO PARES AUTOMÁTICOS DE PERGUNTA/RESPOSTA")
print("=" * 80)

# O curso pode chamar essa etapa de QAGenerationChain.
# Em versões atuais/clássicas do LangChain, a classe também aparece como QAGenerateChain.
#
# Importante:
# Essa geração depende muito da qualidade do LLM.
# Com modelo local pequeno, pode gerar perguntas/respostas mais fracas.
# Mesmo assim, é útil para demonstrar o fluxo técnico sem custo.

try:
    qa_gen_chain = QAGenerateChain.from_llm(llm)

    chunks_para_geracao = chunks[:5]

    qa_gerados = []

    for i, chunk in enumerate(chunks_para_geracao, start=1):
        print(f"\n⚙️ Gerando QA automático para chunk {i}...")

        try:
            resultado = qa_gen_chain.invoke({"doc": chunk.page_content})
            print(resultado)
            qa_gerados.append(resultado)

        except Exception as erro:
            print(f"⚠️ Falha ao gerar QA para chunk {i}: {erro}")

    print("\n✅ Pares automáticos gerados:")
    print(qa_gerados)

except Exception as erro:
    print("⚠️ Não foi possível executar QAGenerateChain neste ambiente.")
    print("Como alternativa, o dataset manual foi utilizado para avaliação.")
    print(f"Erro: {erro}")


# ============================================================
# 14. ANÁLISE FINAL
# ============================================================

print("\n" + "=" * 80)
print("📌 ANÁLISE FINAL")
print("=" * 80)

acc_no_rag = compute_accuracy(results_no_rag)
acc_rag = compute_accuracy(results_rag)

print(f"Acurácia sem RAG: {acc_no_rag:.2%}")
print(f"Acurácia com RAG: {acc_rag:.2%}")

if acc_rag > acc_no_rag:
    print("✅ O pipeline com RAG apresentou melhor desempenho na avaliação automatizada.")
elif acc_rag == acc_no_rag:
    print("⚠️ O pipeline com RAG apresentou desempenho semelhante ao sem RAG.")
else:
    print("⚠️ O pipeline com RAG apresentou desempenho inferior nesta avaliação.")

print("""
Observações:
- A avaliação automatizada com QAEvalChain depende da qualidade do LLM avaliador.
- Como usamos modelo local por decisão de custo zero, os julgamentos podem não ser tão estáveis quanto com modelos maiores.
- Ainda assim, a atividade demonstra o fluxo completo:
  dataset → geração de respostas → comparação com gabarito → métrica de acurácia.
- Ajustes possíveis:
  - alterar k do retriever;
  - alterar chunk_size e chunk_overlap;
  - testar outro modelo local no Ollama;
  - melhorar o dataset de avaliação;
  - usar embeddings diferentes;
  - persistir os resultados em JSON/CSV.
""")

print("\n✅ ATIVIDADE DE AVALIAÇÃO RAG FINALIZADA.")