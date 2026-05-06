# ? GUIA DE EXECUÇÃO - Pipeline RAG Completo

## ? STATUS DA CONFIGURAÇÃO

### Modelos Ollama Necessários:
- ? `bge-m3` - Embeddings (downloading...)
- ? `llama3.2:3b` - LLM (downloading...)

### Dependências Python:
- ? `langchain` - Instalado
- ? `langchain-community` - Instalado
- ? `langchain-ollama` - Instalado
- ? `pypdf` - Instalado
- ? `faiss-cpu` - Instalado
- ? `transformers` - Instalado
- ? `sentencepiece` - Instalado

### PDFs Encontrados:
- ? 7 arquivos PDF na pasta `./pdfs/`

---

## ? O QUE FOI CORRIGIDO DO NOTEBOOK ORIGINAL

### 1. **Modelo LLM Corrigido**
```python
# Antes (não existe):
LLM_MODEL = "gemma3:4b"

# Depois (correto):
LLM_MODEL = "llama3.2:3b"
```

### 2. **Warnings Suprimidos**
```python
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['USER_AGENT'] = 'RAGProject/1.0'
```

### 3. **Path Management Melhorado**
```python
from pathlib import Path

# Usa Path objects ao invés de strings
source_name = Path(doc.metadata.get('source')).name
```

### 4. **Logging Estruturado**
- Prints organizados com emojis
- Separadores visuais
- Progresso claro de cada etapa

---

## ? COMO EXECUTAR

### Opção 1: Executar Script Completo

```powershell
cd C:\ia\rag
.\.venv\Scripts\Activate.ps1
python rag_pipeline_completo.py
```

**Tempo estimado:** 5-15 minutos (dependendo da quantidade de PDFs)

### Opção 2: Executar Notebook Jupyter

```powershell
cd C:\ia\rag
.\.venv\Scripts\Activate.ps1
jupyter notebook rag_pdfs_ollama_faiss_langchain.ipynb
```

---

## ? VERIFICAR STATUS DOS DOWNLOADS

```powershell
# Ver modelos instalados
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list

# Testar modelo de embeddings
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run bge-m3

# Testar modelo LLM
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run llama3.2:3b
```

---

## ? ESTRUTURA DO PIPELINE

### Etapas Implementadas:

1. ? **Carregamento de PDFs** (DirectoryLoader + PyPDFLoader)
2. ? **Chunking com Tokenizer HuggingFace** (BAAI/bge-m3)
3. ? **Embeddings Locais** (Ollama + bge-m3)
4. ? **Vector Store FAISS** (persistência em disco)
5. ? **Retriever** (similarity search, k=4)
6. ? **LLM Local** (Ollama + llama3.2:3b)
7. ? **RAG Chain Básica** (LangChain)
8. ? **Query Rewriting** (refinamento de consulta)
9. ? **MultiQuery Retriever** (múltiplas variações)
10. ? **HyDE** (Hypothetical Document Embeddings)
11. ? **Comparação de Técnicas**

---

## ? TÉCNICAS AVANÇADAS IMPLEMENTADAS

### 1. Query Rewriting
**O que faz:** Reformula a pergunta para melhorar a busca  
**Benefício:** Elimina ambiguidades, preserva termos técnicos

### 2. MultiQuery Retriever
**O que faz:** Gera variações da pergunta e busca com todas  
**Benefício:** Maior cobertura, menos dependência de palavras exatas

### 3. HyDE (Hypothetical Document Embeddings)
**O que faz:** LLM gera resposta hipotética ? busca por ela  
**Benefício:** Busca semântica mais próxima do conteúdo real

---

## ? ARQUIVOS GERADOS

```
C:\ia\rag\
??? rag_pipeline_completo.py       ? Script Python executável
??? rag_pdfs_ollama_faiss_langchain.ipynb  ? Notebook original
??? pdfs/                           ? Seus PDFs (7 arquivos)
??? vectorstore_faiss/              ? Índice FAISS (gerado após execução)
    ??? index.faiss
    ??? index.pkl
```

---

## ? TROUBLESHOOTING

### Erro: "Model 'bge-m3' not found"
**Solução:** Aguarde o download terminar ou execute:
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull bge-m3
```

### Erro: "Model 'llama3.2:3b' not found"
**Solução:** Aguarde o download terminar ou execute:
```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.2:3b
```

### Erro: "No module named 'transformers'"
**Solução:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install transformers sentencepiece
```

### Erro: "Nenhum PDF encontrado"
**Solução:** Os PDFs já estão na pasta! Verifique se o script está rodando da pasta correta:
```powershell
cd C:\ia\rag
python rag_pipeline_completo.py
```

---

## ? COMPARAÇÃO DE RESULTADOS

Ao final da execução, você verá uma comparação das 4 técnicas:

```
? COMPARAÇÃO FINAL DAS TÉCNICAS
================================================================================

1?? CONSULTA ORIGINAL:
[Resposta baseada na busca direta]

2?? QUERY REWRITING:
[Resposta com consulta refinada]

3?? MULTIQUERY:
[Resposta com múltiplas variações da consulta]

4?? HYDE:
[Resposta usando documento hipotético]
```

**Análise:** Compare qual técnica trouxe:
- ? Resposta mais precisa
- ? Documentos mais relevantes
- ? Melhor cobertura do assunto

---

## ? PRÓXIMOS PASSOS

### Após executar com sucesso:

1. **Testar com perguntas variadas:**
   ```python
   perguntas = [
       "Quais são as principais regras do GTB?",
       "Como funciona o fluxo de melhor oferta?",
       "Qual a diferença entre as categorias do GTB?",
   ]
   ```

2. **Adicionar mais PDFs:**
   - Coloque novos PDFs em `./pdfs/`
   - Re-execute o script
   - O FAISS será recriado automaticamente

3. **Ajustar parâmetros:**
   ```python
   CHUNK_SIZE = 700      # Teste: 500, 1000
   CHUNK_OVERLAP = 120   # Teste: 50, 200
   search_kwargs={"k": 4}  # Teste: k=2, k=6
   temperature=0.2       # Teste: 0.0 (deterministico), 0.5
   ```

4. **Implementar reranking:**
   - Adicionar cross-encoder para reordenar resultados
   - Melhorar precisão final

5. **Criar interface de chat:**
   - Gradio ou Streamlit
   - Conversação com histórico
   - Upload de PDFs dinâmico

---

## ? DICAS PRO

### Reutilizar FAISS sem reprocessar:
```python
# Se o índice já existe, apenas carregue:
if VECTOR_DB_DIR.exists():
    vectorstore = FAISS.load_local(...)
else:
    vectorstore = FAISS.from_documents(...)
    vectorstore.save_local(...)
```

### Monitorar uso de memória:
```python
import psutil
print(f"RAM: {psutil.Process().memory_info().rss / 1024 / 1024:.0f} MB")
```

### Logging detalhado:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## ? RECURSOS ADICIONAIS

### Papers:
- **RAG:** "Retrieval-Augmented Generation" (Lewis et al., 2020)
- **HyDE:** "Precise Zero-Shot Dense Retrieval" (Gao et al., 2022)
- **MultiQuery:** LangChain Documentation

### Modelos Alternativos:
- **Embeddings:** `nomic-embed-text`, `mxbai-embed-large`
- **LLM:** `llama3.2:1b` (mais rápido), `llama3.1:8b` (melhor qualidade)

---

## ? CHECKLIST DE EXECUÇÃO

- [ ] Modelos Ollama baixados (bge-m3 + llama3.2:3b)
- [ ] Dependências Python instaladas
- [ ] PDFs na pasta `./pdfs/`
- [ ] Ambiente virtual ativado
- [ ] Script executado com sucesso
- [ ] VectorStore FAISS criado
- [ ] Comparação de técnicas visualizada

---

**? Seu pipeline RAG está pronto para uso profissional!**

Qualquer dúvida, consulte este guia ou os comentários no código.

