# ============================================================
# ATIVIDADE - DOCLOADERS, CHUNKING, EMBEDDINGS E VECTOR STORE
# ============================================================

# Objetivo:
# Implementar diferentes formas de carregamento de documentos com LangChain,
# aplicar estratégias de chunking, gerar embeddings, armazenar os chunks
# em uma vector store e validar buscas por similaridade.
#
# Observação importante:
# O curso originalmente utiliza OpenAI Embeddings, ChatOpenAI, LangSmith e,
# em alguns momentos, serviços externos como Pinecone.
#
# Neste ambiente, a OpenAI retornou erro 429 - insufficient_quota,
# e o LangSmith retornou erro 403 - Forbidden.
#
# Por isso, esta atividade mantém a arquitetura proposta, mas usa alternativas
# sem custo:
# - HuggingFaceEmbeddings no lugar de OpenAIEmbeddings
# - FAISS/InMemory no lugar de serviços pagos
# - Pinecone deixado apenas como bloco comentado/opcional
# ============================================================


# ============================================================
# IMPORTAÇÕES
# ============================================================

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
)

from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. TEXT LOADER - ARQUIVOS TXT
# ============================================================

print("\n" + "=" * 80)
print("1. CARREGANDO DOCUMENTO TXT COM TextLoader")
print("=" * 80)

txt_loader = TextLoader(
    "dados/politica_cartao.txt",
    encoding="utf-8"
)

documentos_txt = txt_loader.load()

print(f"Documentos TXT carregados: {len(documentos_txt)}")
print("Prévia do conteúdo TXT:")
print(documentos_txt[0].page_content[:300])


# ============================================================
# 2. PDF LOADER - ARQUIVOS PDF COM METADADOS
# ============================================================

print("\n" + "=" * 80)
print("2. CARREGANDO DOCUMENTO PDF COM PyPDFLoader")
print("=" * 80)

pdf_loader = PyPDFLoader("dados/manual_compras.pdf")

documentos_pdf = pdf_loader.load()

print(f"Páginas carregadas do PDF: {len(documentos_pdf)}")

for i, doc in enumerate(documentos_pdf, start=1):
    print(f"\n--- Página {i} ---")
    print("Metadados:")
    print(doc.metadata)
    print("Conteúdo:")
    print(doc.page_content[:300])


# ============================================================
# 3. WEB LOADER - PÁGINA WEB
# ============================================================

print("\n" + "=" * 80)
print("3. CARREGANDO CONTEÚDO WEB COM WebBaseLoader")
print("=" * 80)

# O WebBaseLoader depende de conexão com a internet.
# Se sua rede bloquear acesso externo, esta etapa pode falhar.
# Nesse caso, basta comentar este bloco e seguir a atividade.

try:
    web_loader = WebBaseLoader("https://docs.python.org/3/tutorial/index.html")
    documentos_web = web_loader.load()

    print(f"Documentos web carregados: {len(documentos_web)}")
    print("Metadados da página:")
    print(documentos_web[0].metadata)
    print("Prévia do conteúdo web:")
    print(documentos_web[0].page_content[:300])

except Exception as erro:
    print("Não foi possível carregar a página web.")
    print(f"Erro: {erro}")
    documentos_web = []


# ============================================================
# 4. DIRECTORY LOADER - DIRETÓRIO DE DOCUMENTOS TXT
# ============================================================

print("\n" + "=" * 80)
print("4. CARREGANDO DIRETÓRIO COM DirectoryLoader")
print("=" * 80)

directory_loader = DirectoryLoader(
    path="dados/diretorio",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documentos_diretorio = directory_loader.load()

print(f"Documentos carregados do diretório: {len(documentos_diretorio)}")

for doc in documentos_diretorio:
    print("\nArquivo:")
    print(doc.metadata)
    print(doc.page_content[:200])


# ============================================================
# 5. MERGE LOADER - UNINDO FONTES DIFERENTES
# ============================================================

print("\n" + "=" * 80)
print("5. UNINDO DIFERENTES FONTES COM MergedDataLoader")
print("=" * 80)

# O MergedDataLoader permite combinar loaders diferentes em uma única coleção.
# Aqui unimos TXT, PDF e DirectoryLoader.
#
# A parte web foi mantida fora do merge por poder depender da conexão externa,
# mas poderia entrar também se necessário.

merge_loader = MergedDataLoader(
    loaders=[
        txt_loader,
        pdf_loader,
        directory_loader
    ]
)

documentos_unificados = merge_loader.load()

# Se o WebBaseLoader funcionou, adicionamos os documentos web manualmente.
documentos_unificados.extend(documentos_web)

print(f"Total de documentos unificados: {len(documentos_unificados)}")

for i, doc in enumerate(documentos_unificados, start=1):
    print(f"\n--- Documento {i} ---")
    print("Metadados:")
    print(doc.metadata)
    print("Prévia:")
    print(doc.page_content[:200])


# ============================================================
# 6. CHUNKING SIMPLES POR CONTAGEM DE CARACTERES
# ============================================================

print("\n" + "=" * 80)
print("6. CHUNKING POR CARACTERES")
print("=" * 80)

character_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks_caracteres = character_splitter.split_documents(documentos_unificados)

print(f"Total de chunks por caracteres: {len(chunks_caracteres)}")

for i, chunk in enumerate(chunks_caracteres[:5], start=1):
    print(f"\n--- Chunk caracteres {i} ---")
    print(chunk.page_content[:300])


# ============================================================
# 7. CHUNKING RECURSIVO POR CARACTERES
# ============================================================

print("\n" + "=" * 80)
print("7. CHUNKING COM RecursiveCharacterTextSplitter")
print("=" * 80)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks_recursivos = recursive_splitter.split_documents(documentos_unificados)

print(f"Total de chunks recursivos: {len(chunks_recursivos)}")

for i, chunk in enumerate(chunks_recursivos[:5], start=1):
    print(f"\n--- Chunk recursivo {i} ---")
    print(chunk.page_content[:300])


# ============================================================
# 8. CHUNKING POR TOKENS COM TIKTOKEN
# ============================================================

print("\n" + "=" * 80)
print("8. CHUNKING POR TOKENS COM TIKTOKEN")
print("=" * 80)

# O curso utiliza Tiktoken porque ele é compatível com os modelos da OpenAI.
# Mesmo sem usar OpenAI por falta de quota, esta etapa é útil para entender
# divisão por tokens.
#
# Observação:
# O nome correto da biblioteca é tiktoken, não TickToken.

token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=300,
    chunk_overlap=50
)

chunks_tokens = token_splitter.split_documents(documentos_unificados)

print(f"Total de chunks por tokens: {len(chunks_tokens)}")

for i, chunk in enumerate(chunks_tokens[:5], start=1):
    print(f"\n--- Chunk token {i} ---")
    print(chunk.page_content[:300])


# ============================================================
# 9. EMBEDDINGS COM HUGGING FACE - CAMINHO ALTERNATIVO SEM OPENAI
# ============================================================

print("\n" + "=" * 80)
print("9. CONFIGURANDO EMBEDDINGS COM HUGGING FACE")
print("=" * 80)

# Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
# Em vez de usar OpenAIEmbeddings, utilizamos HuggingFaceEmbeddings localmente.
#
# Escolhemos um modelo multilíngue porque nossos documentos estão em português.
# Isso tende a ser mais adequado do que usar embeddings focados apenas em inglês.

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small"
)

print("Embeddings Hugging Face configurados com sucesso.")

# ------------------------------------------------------------
# Como seria no caminho original com OpenAI:
# ------------------------------------------------------------
# from langchain_openai import OpenAIEmbeddings
#
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )


# ============================================================
# 10. CHUNKING SEMÂNTICO COM HUGGING FACE
# ============================================================

print("\n" + "=" * 80)
print("10. CHUNKING SEMÂNTICO COM EMBEDDINGS")
print("=" * 80)

try:
    from langchain_experimental.text_splitter import SemanticChunker

    # Este trecho é um caminho alternativo ao OpenAI por conta de falta de crédito/quota.
    # O curso pode usar embeddings da OpenAI para chunking semântico.
    # Aqui usamos os embeddings locais do Hugging Face.

    semantic_splitter = SemanticChunker(
        embeddings=embeddings
    )

    chunks_semanticos = semantic_splitter.split_documents(documentos_unificados)

    print(f"Total de chunks semânticos: {len(chunks_semanticos)}")

    for i, chunk in enumerate(chunks_semanticos[:5], start=1):
        print(f"\n--- Chunk semântico {i} ---")
        print(chunk.page_content[:300])

except Exception as erro:
    print("Não foi possível executar o chunking semântico.")
    print("Isso pode ocorrer por incompatibilidade de versão ou dependência.")
    print(f"Erro: {erro}")
    chunks_semanticos = chunks_recursivos


# ============================================================
# 11. VECTOR STORE IN-MEMORY
# ============================================================

print("\n" + "=" * 80)
print("11. ARMAZENANDO CHUNKS EM InMemoryVectorStore")
print("=" * 80)

# Para manter a atividade simples e sem custo, usamos o InMemoryVectorStore.
# Ele é suficiente para estudo, testes e documentos pequenos.

vector_store_memory = InMemoryVectorStore.from_documents(
    documents=chunks_recursivos,
    embedding=embeddings
)

retriever_memory = vector_store_memory.as_retriever(
    search_kwargs={"k": 3}
)

print("Vector store em memória configurada com sucesso.")


# ============================================================
# 12. VECTOR STORE COM FAISS
# ============================================================

print("\n" + "=" * 80)
print("12. ARMAZENANDO CHUNKS COM FAISS")
print("=" * 80)

# FAISS é uma alternativa local, sem custo de API, muito usada para busca vetorial.
# Diferente do InMemoryVectorStore, ele é mais apropriado para experimentos maiores
# e pode ser persistido em disco se necessário.

vector_store_faiss = FAISS.from_documents(
    documents=chunks_recursivos,
    embedding=embeddings
)

print("Vector store FAISS configurada com sucesso.")


# ============================================================
# 13. BUSCA POR SIMILARIDADE COM SCORES
# ============================================================

print("\n" + "=" * 80)
print("13. VALIDANDO BUSCA POR SIMILARIDADE COM SCORES")
print("=" * 80)

perguntas = [
    "O cartão corporativo pode ser usado para compras pessoais?",
    "Como funciona o reembolso de despesas?",
    "O que é necessário para cadastrar fornecedores?",
    "Viagens corporativas precisam de aprovação?",
]

todos_scores = []

for pergunta in perguntas:
    print("\n" + "-" * 80)
    print(f"Pergunta: {pergunta}")

    resultados = vector_store_faiss.similarity_search_with_score(
        query=pergunta,
        k=3
    )

    for i, (doc, score) in enumerate(resultados, start=1):
        todos_scores.append(score)
        source_name = Path(doc.metadata.get('source', 'unknown')).name

        print(f"\nResultado {i}")
        print(f"Score de similaridade/distância: {score:.3f}")
        print(f"Fonte: {source_name}")
        print("Trecho recuperado:")
        print(doc.page_content[:300] + "...")

# Análise dos scores
print("\n" + "=" * 80)
print("📊 ANÁLISE DE QUALIDADE DAS BUSCAS")
print("=" * 80)
print(f"Score médio: {sum(todos_scores)/len(todos_scores):.3f}")
print(f"Melhor score (mais similar): {min(todos_scores):.3f}")
print(f"Pior score (menos similar): {max(todos_scores):.3f}")
print("\n💡 No FAISS com L2: Scores menores = maior similaridade")
print(f"   Qualidade geral: {'Excelente ⭐⭐⭐⭐⭐' if min(todos_scores) < 0.2 else 'Boa ⭐⭐⭐⭐' if min(todos_scores) < 0.3 else 'Regular ⭐⭐⭐'}")


# ============================================================
# 14. BUSCA COM FILTRO POR METADADOS
# ============================================================

print("\n" + "=" * 80)
print("14. BUSCA COM FILTRO POR METADADOS")
print("=" * 80)

# A busca com filtro depende da vector store e da estrutura dos metadados.
# No FAISS local do LangChain, o filtro simples por metadados pode ser usado
# em alguns cenários.
#
# Aqui buscamos apenas documentos cuja fonte contenha manual_compras.pdf.

try:
    resultados_filtrados = vector_store_faiss.similarity_search(
        query="fornecedores e nota fiscal",
        k=3,
        filter={"source": "dados/manual_compras.pdf"}
    )

    print(f"Resultados filtrados encontrados: {len(resultados_filtrados)}")

    for i, doc in enumerate(resultados_filtrados, start=1):
        print(f"\nResultado filtrado {i}")
        print(doc.metadata)
        print(doc.page_content[:500])

except Exception as erro:
    print("A busca com filtro não funcionou nesta vector store/localidade.")
    print("Mesmo assim, a busca por similaridade com score foi validada.")
    print(f"Erro: {erro}")


# ============================================================
# 15. PINECONE - BLOCO OPCIONAL COMENTADO
# ============================================================

# A atividade cita integração com banco vetorial na nuvem, como Pinecone.
# Como nossa regra nesta formação é não gastar nenhum real e evitar dependência
# de serviços pagos, este bloco fica apenas como referência conceitual.
#
# Para usar Pinecone, seria necessário:
# - Criar conta no Pinecone
# - Gerar API Key
# - Criar/configurar índice
# - Instalar langchain-pinecone
#
# Comandos:
# pip install langchain-pinecone pinecone
#
# Exemplo conceitual:
#
# from langchain_pinecone import PineconeVectorStore
#
# vector_store_pinecone = PineconeVectorStore.from_documents(
#     documents=chunks_recursivos,
#     embedding=embeddings,
#     index_name="nome-do-indice"
# )
#
# retriever_pinecone = vector_store_pinecone.as_retriever(
#     search_kwargs={"k": 3}
# )
#
# Nesta entrega, utilizei InMemoryVectorStore e FAISS como alternativas locais.


# ============================================================
# FIM DA ATIVIDADE
# ============================================================

print("\n" + "=" * 80)
print("ATIVIDADE FINALIZADA COM SUCESSO")
print("=" * 80)

print("""
Resumo do que foi implementado:

- TextLoader para arquivos TXT
- PyPDFLoader para PDFs com metadados
- WebBaseLoader para páginas web
- DirectoryLoader para múltiplos arquivos
- MergedDataLoader para unir fontes diferentes
- CharacterTextSplitter para chunking por caracteres
- RecursiveCharacterTextSplitter para chunking recursivo
- CharacterTextSplitter com tiktoken para chunking por tokens
- SemanticChunker com HuggingFaceEmbeddings como alternativa ao OpenAI
- InMemoryVectorStore
- FAISS local
- Busca por similaridade com scores
- Tentativa de busca com filtros
- Pinecone documentado como opção em nuvem, mas não usado por decisão de custo zero
""")