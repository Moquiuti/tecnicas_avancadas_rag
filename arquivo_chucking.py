# ============================================================
# CARREGAMENTO DO DOCUMENTO + CHUNKING
# ============================================================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Carrega o arquivo de texto
loader = TextLoader("politica_cartao.txt", encoding="utf-8")
documentos = loader.load()

print(f"Quantidade de documentos carregados: {len(documentos)}")

# Divide o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documentos)

print(f"Quantidade de chunks criados: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)