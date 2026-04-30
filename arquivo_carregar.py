# ============================================================
# CARREGAMENTO DO DOCUMENTO
# ============================================================

from langchain_community.document_loaders import TextLoader

# Este é o carregamento básico do documento, conforme o fluxo do curso.
# O TextLoader lê um arquivo de texto e transforma o conteúdo em documento LangChain.

loader = TextLoader("politica_cartao.txt", encoding="utf-8")
documentos = loader.load()

print(f"Quantidade de documentos carregados: {len(documentos)}")
print(documentos[0].page_content)