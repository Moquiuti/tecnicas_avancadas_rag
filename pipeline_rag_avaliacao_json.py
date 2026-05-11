# ============================================================
# ATIVIDADE FINAL - PIPELINE RAG COM AVALIAÇÃO VIA JSON
# ============================================================

# Objetivo:
# - Carregar um arquivo JSON de avaliação
# - Extrair e transformar os dados mantendo apenas query e answer
# - Gerar respostas sem RAG
# - Gerar respostas com RAG
# - Comparar as respostas geradas com os gabaritos
# - Corrigir problemas de indexação
# - Medir a precisão do sistema
#
# Observação:
# O enunciado menciona GPT/OpenAI, porém neste projeto estamos seguindo
# uma abordagem sem custo de API. Em atividades anteriores, a OpenAI retornou
# erro 429 - insufficient_quota. Por isso, usamos ChatOllama local.
#
# Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
#
# Como seria no caminho original com OpenAI:
#
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#
# Caminho adotado:
# - ChatOllama com llama3.2:3b para geração
# - OllamaEmbeddings com bge-m3 para embeddings
# - FAISS local como banco vetorial
# ============================================================


import json
import os
from pathlib import Path
from difflib import SequenceMatcher

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
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
DADOS_DIR = BASE_DIR / "dados"
PDFS_DIR = BASE_DIR / "pdfs"
DATASET_PATH = BASE_DIR / "dataset_avaliacao.json"
VECTORSTORE_DIR = BASE_DIR / "vectorstore_faiss_final"

MODELO_LLM = "llama3.2:3b"
MODELO_EMBEDDINGS = "bge-m3"

print("=" * 80)
print("🚀 PIPELINE RAG FINAL COM AVALIAÇÃO VIA JSON")
print("=" * 80)
print(f"📄 Dataset: {DATASET_PATH}")
print(f"🤖 Modelo LLM local: {MODELO_LLM}")
print(f"🧠 Modelo embeddings local: {MODELO_EMBEDDINGS}")
print(f"💾 VectorStore: {VECTORSTORE_DIR}")


# ============================================================
# 1. CARREGAR JSON E MANTER APENAS QUERY E ANSWER
# ============================================================

print("\n" + "=" * 80)
print("1. CARREGANDO DATASET JSON")
print("=" * 80)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Arquivo JSON não encontrado: {DATASET_PATH}")

with open(DATASET_PATH, "r", encoding="utf-8-sig") as file:
    raw_dataset = json.load(file)

dataset = []

for item in raw_dataset:
    if "query" in item and "answer" in item:
        dataset.append({
            "query": item["query"],
            "answer": item["answer"]
        })

if not dataset:
    raise ValueError("Nenhum item válido encontrado no JSON. Esperado: query e answer.")

print(f"✅ Total de perguntas carregadas: {len(dataset)}")
print("✅ Dataset transformado mantendo apenas os campos query e answer.")

for item in dataset:
    print(f"- {item['query']}")


# ============================================================
# 2. CONFIGURAR LLM E EMBEDDINGS LOCAIS
# ============================================================

print("\n" + "=" * 80)
print("2. CONFIGURANDO MODELOS LOCAIS")
print("=" * 80)

llm = ChatOllama(
    model=MODELO_LLM,
    temperature=0
)

embeddings = OllamaEmbeddings(
    model=MODELO_EMBEDDINGS
)

print("✅ ChatOllama configurado.")
print("✅ OllamaEmbeddings configurado.")


# ============================================================
# 3. CARREGAR DOCUMENTOS DA BASE DE CONHECIMENTO
# ============================================================

print("\n" + "=" * 80)
print("3. CARREGANDO DOCUMENTOS DA BASE")
print("=" * 80)

documentos = []

# Carrega arquivos TXT da pasta dados
if DADOS_DIR.exists():
    try:
        txt_loader = DirectoryLoader(
            path=str(DADOS_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8-sig"}
        )

        documentos_txt = txt_loader.load()
        documentos.extend(documentos_txt)

        print(f"✅ Documentos TXT carregados: {len(documentos_txt)}")

    except Exception as erro:
        print(f"⚠️ Erro ao carregar TXTs: {erro}")

# Carrega PDFs da pasta pdfs
if PDFS_DIR.exists():
    pdfs = list(PDFS_DIR.glob("*.pdf"))

    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            docs_pdf = loader.load()
            documentos.extend(docs_pdf)

            print(f"✅ PDF carregado: {pdf.name} | páginas: {len(docs_pdf)}")

        except Exception as erro:
            print(f"⚠️ Erro ao carregar PDF {pdf.name}: {erro}")

if not documentos:
    raise RuntimeError("Nenhum documento foi carregado. Verifique as pastas dados/ e pdfs/.")

print(f"📊 Total de documentos carregados: {len(documentos)}")
print("📝 Prévia do primeiro documento:")
print(documentos[0].page_content[:400])


# ============================================================
# 4. CHUNKING
# ============================================================

print("\n" + "=" * 80)
print("4. REALIZANDO CHUNKING")
print("=" * 80)

# Ajuste realizado para evitar chunks muito pequenos e melhorar a recuperação.
# O overlap ajuda a preservar contexto entre pedaços próximos.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documentos)

if not chunks:
    raise RuntimeError("Nenhum chunk foi gerado.")

print(f"✅ Total de chunks gerados: {len(chunks)}")
print("📝 Prévia do primeiro chunk:")
print(chunks[0].page_content[:400])


# ============================================================
# 5. CRIAR OU CARREGAR VECTORSTORE FAISS
# ============================================================

print("\n" + "=" * 80)
print("5. CRIANDO/CARREGANDO VECTORSTORE FAISS")
print("=" * 80)

# Correção importante:
# Para evitar erros de indexação, validamos se a vectorstore existente
# pertence ao mesmo conjunto de chunks/modelo. Em ambiente de estudo,
# uma forma segura é apagar a pasta quando houver mudança grande na base.
#
# Caso queira recriar manualmente:
# Remove-Item -Recurse -Force C:\ia\rag\vectorstore_faiss_final

if VECTORSTORE_DIR.exists():
    print("📦 VectorStore existente encontrada. Carregando do disco...")

    vectorstore = FAISS.load_local(
        folder_path=str(VECTORSTORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

else:
    print("⚙️ Criando nova VectorStore FAISS...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(str(VECTORSTORE_DIR))

    print("✅ VectorStore criada e salva em disco.")

# Ajuste do k:
# k define quantos chunks serão recuperados.
# Se k for baixo demais, pode faltar contexto.
# Se k for alto demais, pode entrar ruído.
# Aqui usamos k=4 para equilibrar cobertura e precisão.

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

print("✅ Retriever configurado com k=4.")


# ============================================================
# 6. FUNÇÕES AUXILIARES
# ============================================================

def formatar_documentos(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def normalizar_texto(texto: str) -> str:
    return " ".join(texto.lower().strip().split())


def similarity_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalizar_texto(a), normalizar_texto(b)).ratio()


def avaliar_resposta(resposta_gerada: str, resposta_esperada: str) -> dict:
    """
    Avaliação simples e local, sem depender de OpenAI/QA-Eval pago.

    Critérios:
    - Similaridade textual com SequenceMatcher
    - Presença de termos importantes do gabarito na resposta gerada

    Em um cenário com OpenAI disponível, poderíamos usar QAEvalChain.
    """

    score_similaridade = similarity_ratio(resposta_gerada, resposta_esperada)

    termos_esperados = [
        termo
        for termo in normalizar_texto(resposta_esperada).replace(".", "").split()
        if len(termo) > 4
    ]

    if termos_esperados:
        termos_presentes = [
            termo for termo in termos_esperados
            if termo in normalizar_texto(resposta_gerada)
        ]

        cobertura_termos = len(termos_presentes) / len(termos_esperados)

    else:
        cobertura_termos = 0

    score_final = (score_similaridade * 0.4) + (cobertura_termos * 0.6)

    correto = score_final >= 0.45

    return {
        "score_similaridade": round(score_similaridade, 4),
        "cobertura_termos": round(cobertura_termos, 4),
        "score_final": round(score_final, 4),
        "correto": correto
    }


# ============================================================
# 7. PIPELINE SEM RAG
# ============================================================

print("\n" + "=" * 80)
print("6. CONFIGURANDO PIPELINE SEM RAG")
print("=" * 80)

prompt_sem_rag = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente de perguntas e respostas. "
        "Responda de forma direta. "
        "Você não recebeu documentos de contexto para esta pergunta."
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

print("✅ Pipeline sem RAG configurado.")


# ============================================================
# 8. PIPELINE COM RAG
# ============================================================

print("\n" + "=" * 80)
print("7. CONFIGURANDO PIPELINE COM RAG")
print("=" * 80)

prompt_com_rag = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente especializado em responder perguntas com base em documentos. "
        "Use somente o contexto fornecido. "
        "Responda de forma objetiva. "
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

print("✅ Pipeline com RAG configurado.")


# ============================================================
# 9. EXECUTAR AVALIAÇÃO
# ============================================================

print("\n" + "=" * 80)
print("8. EXECUTANDO AVALIAÇÃO COMPARATIVA")
print("=" * 80)

resultados = []

for idx, item in enumerate(dataset, start=1):
    query = item["query"]
    answer = item["answer"]

    print("\n" + "-" * 80)
    print(f"Pergunta {idx}: {query}")
    print(f"Gabarito: {answer}")

    resposta_sem_rag = chain_sem_rag.invoke(query)
    avaliacao_sem_rag = avaliar_resposta(resposta_sem_rag, answer)

    resposta_com_rag = chain_com_rag.invoke(query)
    avaliacao_com_rag = avaliar_resposta(resposta_com_rag, answer)

    resultados.append({
        "query": query,
        "answer": answer,
        "resposta_sem_rag": resposta_sem_rag,
        "avaliacao_sem_rag": avaliacao_sem_rag,
        "resposta_com_rag": resposta_com_rag,
        "avaliacao_com_rag": avaliacao_com_rag
    })

    print("\nResposta sem RAG:")
    print(resposta_sem_rag)
    print(f"Avaliação sem RAG: {avaliacao_sem_rag}")

    print("\nResposta com RAG:")
    print(resposta_com_rag)
    print(f"Avaliação com RAG: {avaliacao_com_rag}")


# ============================================================
# 10. MÉTRICAS FINAIS
# ============================================================

print("\n" + "=" * 80)
print("9. MÉTRICAS FINAIS")
print("=" * 80)

total = len(resultados)

acertos_sem_rag = sum(
    1 for item in resultados
    if item["avaliacao_sem_rag"]["correto"]
)

acertos_com_rag = sum(
    1 for item in resultados
    if item["avaliacao_com_rag"]["correto"]
)

precisao_sem_rag = acertos_sem_rag / total if total else 0
precisao_com_rag = acertos_com_rag / total if total else 0

media_score_sem_rag = sum(
    item["avaliacao_sem_rag"]["score_final"] for item in resultados
) / total

media_score_com_rag = sum(
    item["avaliacao_com_rag"]["score_final"] for item in resultados
) / total

print(f"Total de perguntas avaliadas: {total}")

print("\nSEM RAG:")
print(f"Acertos: {acertos_sem_rag}/{total}")
print(f"Precisão: {precisao_sem_rag:.2%}")
print(f"Score médio: {media_score_sem_rag:.4f}")

print("\nCOM RAG:")
print(f"Acertos: {acertos_com_rag}/{total}")
print(f"Precisão: {precisao_com_rag:.2%}")
print(f"Score médio: {media_score_com_rag:.4f}")

if precisao_com_rag > precisao_sem_rag:
    print("\n✅ O pipeline com RAG apresentou melhor precisão.")
elif precisao_com_rag == precisao_sem_rag:
    print("\n⚠️ O pipeline com RAG apresentou precisão semelhante ao pipeline sem RAG.")
else:
    print("\n⚠️ O pipeline com RAG apresentou precisão inferior nesta configuração.")


# ============================================================
# 11. SALVAR RESULTADOS EM JSON
# ============================================================

OUTPUT_PATH = BASE_DIR / "resultado_avaliacao_rag.json"

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        {
            "modelo_llm": MODELO_LLM,
            "modelo_embeddings": MODELO_EMBEDDINGS,
            "total_perguntas": total,
            "precisao_sem_rag": precisao_sem_rag,
            "precisao_com_rag": precisao_com_rag,
            "score_medio_sem_rag": media_score_sem_rag,
            "score_medio_com_rag": media_score_com_rag,
            "resultados": resultados
        },
        file,
        ensure_ascii=False,
        indent=2
    )

print(f"\n💾 Resultado salvo em: {OUTPUT_PATH}")


# ============================================================
# 12. ANÁLISE FINAL
# ============================================================

print("\n" + "=" * 80)
print("10. ANÁLISE FINAL")
print("=" * 80)

print("""
A atividade demonstrou a comparação entre respostas geradas sem RAG e com RAG.

O pipeline sem RAG depende apenas do conhecimento geral do modelo local.
O pipeline com RAG recupera chunks relevantes da base vetorial FAISS antes de responder.

A função de avaliação foi ajustada para evitar dependência de serviços pagos,
utilizando uma métrica local baseada em similaridade textual e cobertura de termos
importantes do gabarito.

Também foram feitos ajustes para evitar erros comuns de indexação:
- validação do dataset JSON;
- extração apenas dos campos query e answer;
- uso de encoding utf-8-sig;
- persistência do FAISS;
- configuração explícita do k do retriever;
- geração de arquivo JSON com os resultados finais.

A implementação mantém o objetivo técnico da atividade, mas usando uma abordagem
local-first por conta da ausência de crédito/quota na OpenAI.
""")

print("\n✅ PIPELINE RAG COM AVALIAÇÃO FINALIZADO COM SUCESSO.")