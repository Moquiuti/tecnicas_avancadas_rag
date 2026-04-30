# 🤖 Projeto RAG - Retrieval-Augmented Generation

Sistema de IA local usando Ollama e LangChain para implementação de RAG (Retrieval-Augmented Generation).

## 📋 Sobre o Projeto

Este projeto permite executar modelos de linguagem **localmente**, sem custos de API, usando:
- **Ollama** - Para rodar LLMs localmente
- **LangChain** - Framework para construir aplicações com LLMs
- **RAG** - Técnica para melhorar respostas com contexto específico

## ✅ Requisitos

- Python 3.10+
- Ollama instalado
- 8GB+ RAM (16GB recomendado para modelos maiores)

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone <seu-repositorio>
cd rag
```

### 2. Crie o ambiente virtual
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente
```bash
# Copie o exemplo e edite com suas chaves
copy .env.example .env
```

Edite `.env` e adicione suas chaves de API (se usar OpenAI/LangSmith).

### 5. Instale o Ollama
- Download: https://ollama.com/download
- Ou via WinGet: `winget install Ollama.Ollama`

### 6. Baixe um modelo
```bash
ollama pull llama3.2:1b    # Modelo pequeno (1.3 GB)
ollama pull llama3.2:3b    # Modelo melhor (2 GB)
```

## 🧪 Testes Disponíveis

### Teste simples com Ollama
```bash
python teste_ollama.py
```

### Sistema RAG completo
```bash
python exemplo_rag_completo.py
```

### Múltiplas perguntas
```bash
python teste_multiplas_perguntas.py
```

### Setup automático
```bash
python setup_completo.py
```

## 📁 Estrutura do Projeto

```
rag/
├── .env                          # Variáveis de ambiente (não commitado)
├── .gitignore                    # Arquivos ignorados pelo Git
├── requirements.txt              # Dependências Python
├── teste_ollama.py               # Teste simples
├── exemplo_rag_completo.py       # Exemplo RAG real
├── teste_multiplas_perguntas.py  # Testes variados
├── setup_completo.py             # Setup automatizado
├── arquivo_*.py                  # Scripts de processamento
├── rag_*.py                      # Scripts RAG
└── docs/                         # Documentação
    ├── GUIA-OLLAMA.md
    ├── PROXIMOS-PASSOS.md
    ├── COMANDOS-RAPIDOS-OLLAMA.md
    └── RESUMO-CONQUISTAS.md
```

## 🎯 Exemplos de Uso

### Exemplo 1: Pergunta Simples
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:1b")
resposta = llm.invoke("O que é RAG?")
print(resposta.content)
```

### Exemplo 2: RAG com Contexto
```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2:1b")

template = """Use o contexto abaixo para responder:
Contexto: {contexto}
Pergunta: {pergunta}
Resposta:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

resposta = chain.invoke({
    "contexto": "RAG combina busca e geração de texto.",
    "pergunta": "O que é RAG?"
})
```

## 📚 Documentação

- [Guia do Ollama](GUIA-OLLAMA.md)
- [Próximos Passos](PROXIMOS-PASSOS.md)
- [Comandos Rápidos](COMANDOS-RAPIDOS-OLLAMA.md)
- [Resumo de Conquistas](RESUMO-CONQUISTAS.md)

## 🔧 Comandos Úteis

### Gerenciar Modelos Ollama
```bash
ollama list                    # Listar modelos
ollama pull llama3.2:3b        # Baixar modelo
ollama rm llama3.2:1b          # Remover modelo
ollama run llama3.2:1b         # Testar interativamente
```

### Python
```bash
pip install -r requirements.txt --upgrade  # Atualizar deps
pip list                                   # Listar instalados
```

## 🐛 Troubleshooting

### Erro: "model not found"
- O modelo não foi baixado. Execute: `ollama pull llama3.2:1b`

### Erro: "OpenAI quota exceeded"
- Use o Ollama (local) em vez da OpenAI
- Execute: `python teste_ollama.py`

### Ollama não encontrado
- Verifique se está instalado: `ollama --version`
- Reinicie o terminal após instalação

## 📊 Modelos Recomendados

| Modelo | Tamanho | RAM | Qualidade | Uso |
|--------|---------|-----|-----------|-----|
| llama3.2:1b | 1.3 GB | 2 GB | ⭐⭐ | Testes rápidos |
| llama3.2:3b | 2 GB | 4 GB | ⭐⭐⭐ | Uso geral |
| llama3.1:8b | 4.7 GB | 8 GB | ⭐⭐⭐⭐ | Alta qualidade |
| codellama:7b | 3.8 GB | 6 GB | ⭐⭐⭐⭐ | Programação |

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch: `git checkout -b feature/nova-feature`
3. Commit suas mudanças: `git commit -m 'Adiciona nova feature'`
4. Push: `git push origin feature/nova-feature`
5. Abra um Pull Request

## ⚠️ Segurança

- **NUNCA** faça commit do arquivo `.env`
- **NUNCA** compartilhe suas chaves de API
- O `.gitignore` está configurado para proteger dados sensíveis

## 📝 Licença

Este projeto é de código aberto para fins educacionais.

## 🙏 Agradecimentos

- [Ollama](https://ollama.com/) - Por tornar IA local acessível
- [LangChain](https://python.langchain.com/) - Framework incrível
- [Meta AI](https://ai.meta.com/) - Modelos Llama

## 📞 Suporte

- Issues: Abra uma issue neste repositório
- Documentação: Veja os arquivos `.md` na pasta do projeto

---

**Desenvolvido com ❤️ para aprendizado de IA**

