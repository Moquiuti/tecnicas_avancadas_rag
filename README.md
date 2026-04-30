# tecnicas_avancadas_rag

## Observação sobre adaptação da atividade

A atividade original do curso utiliza OpenAI Embeddings, ChatOpenAI e LangSmith para rastreabilidade. Durante a execução local, a chamada para a API da OpenAI retornou erro `429 - insufficient_quota`, indicando ausência de quota/crédito disponível na conta. Além disso, o LangSmith retornou erro `403 - Forbidden`, indicando ausência de permissão ou chave válida para envio dos traces.

Para não interromper a atividade e manter o foco no aprendizado técnico do fluxo RAG, a implementação foi adaptada utilizando:

- `HuggingFaceEmbeddings` no lugar de `OpenAIEmbeddings`;
- `ChatOllama` no lugar de `ChatOpenAI`;
- `InMemoryVectorStore` para armazenamento vetorial em memória;
- LangSmith desabilitado temporariamente no `.env`.

Com isso, foi possível implementar o fluxo principal:

documento → chunking → embeddings → banco vetorial → retriever → prompt → modelo → resposta.

Apesar de o modelo local utilizado não ter retornado respostas satisfatórias em todos os testes, a estrutura técnica da pipeline foi configurada e executada corretamente.
