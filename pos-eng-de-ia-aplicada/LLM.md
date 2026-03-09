# 🌐 Large Language Models (LLMs)

LLMs são modelos de inteligência artificial que **aprendem padrões de linguagem** a partir de grandes quantidades de texto.

Eles podem **gerar texto, responder perguntas, resumir conteúdos, traduzir idiomas e ATÉÉÉÉÉ criar código (oh no)**.

### EM OUTRAS PALAVRAS:

- Lê uma **enorme quantidade de texto** para aprender como as palavras e frases funcionam juntas
- Aprende padrões e relações entre palavras
- Quando você pede algo, ele **tenta prever o próximo token mais provável**
- Com isso, consegue gerar respostas coerentes e contextualizadas

> É como se o modelo tivesse lido milhões de livros e agora adivinhasse o que vem a seguir de forma bem inteligente.

---

# 🧠 1️⃣ O que são LLMs?

- **LLM** = Large Language Model (Modelo de Linguagem de Grande Escala)
- São treinados com **quantidades enormes de texto**, aprendendo padrões de palavras, frases e contexto
- **GPT** significa:
  - **Generative:** gera texto novo a partir de um prompt
  - **Pre-trained:** é pré-treinado com muito conteúdo antes de ser usado
  - **Transformer:** arquitetura de rede neural que processa o texto de forma eficiente

---

# 🔢 2️⃣ Tokenização

Antes de processar o texto, o LLM **divide em tokens**:

- Tokens = unidades menores que palavras
- Podem ser palavras, partes de palavras ou sinais de pontuação
- Exemplo:

```text
Texto: "O gato subiu no telhado."
Tokens: ["O", " gato", " subiu", " no", " telhado", "."]
```

---

# 🧠 3️⃣ Embeddings: representando palavras como vetores

Depois de tokenizar, cada token é **transformado em um vetor numérico**, chamado **embedding**:

- Esse vetor captura **significado, contexto e relações com outras palavras**
- Palavras com contextos semelhantes ficam **próximas no espaço vetorial**

Exemplos:

- `"caneta"` e `"lápis"` → próximos, ambos objetos de escrita
- `"professor"` e `"escola"` → relacionados por contexto, mesmo não sendo sinônimos

---

### 🔹 Relações e analogias

Embeddings permitem **operar com vetores para capturar relações**:

```text
rei - homem + mulher = rainha
verão - quente + frio ≈ inverno

```

- Relações recorrentes se tornam **direções no espaço vetorial** (genero, capital, singular, plural)
- Permite **resolver analogias e entender associação semântica**

---

# 🏗 4️⃣ Transformers e Attention

### 🔹 Transformer

- Arquitetura de rede neural moderna, feita pra processar sequencia de texto
- Permite processar **todos os tokens em paralelo** diferente de modelos antigos que liam sequencialmente
- Usa **Self-Attention** para entender quais tokens são relevantes no contexto

Exemplo:

```text
    "A Julia disse à Carla que ela ganhou o prêmio."
```

- Quem é "ela"? Julia ou Carla?
- O mecanismo **atenta ao contexto completo** para tomar a decisão.

---

### 🔹 Positional Encoding

- Vetores (embeddings) **não carregam ordem por si só**
- O modelo adiciona **informação de posição** para entender sequência
- Assim: `"o cachorro mordeu o homem"` ≠ `"o homem mordeu o cachorro"`

---

# 🎲 5️⃣ Probabilidades e Decoding

Depois que o Transformer processa os embeddings, ele calcula as **probabilidades para os próximos tokens**:

Exemplo:

```text
Prompt: "O dia estava"

Previsão de tokens:
    ensolarado: 40%
    chuvoso: 25%
    nublado: 20%
    frio: 15%
    bonito: 7%
```

- O modelo **escolhe um token com base nessas probabilidades**

### 🔹 Parâmetros de geração

- **Temperature:** controla aleatoriedade
  - Baixa →  previsível    (método greedy)
  - Alta →   aleatorio
- **Top-K:** limita a escolha aos K tokens mais prováveis
- **Top-P (nucleus sampling):** soma probabilidades até um limite (ex: 90%) e escolhe dentro desse conjunto

> Esses parâmetros influenciam **qualidade e variedade das respostas**.

---

# ⏱ 6️⃣ Geração passo a passo (Sampling)

O LLM gera texto **token por token** mas não gera tudo de uma vez, ele vai recalculando contexto a cada passo:

1. Analisa o texto já gerado
2. Calcula a probabilidade dos próximos tokens
3. Escolhe um token
4. Adiciona ao contexto e repete

> Quanto maior o texto, maior o custo computacional.

---

# 🔎 7️⃣ Conceitos-chave de aprendizado

- **Similaridade:** palavras parecidas ou usadas em contextos semelhantes ficam próximas
- **Associação:** palavras relacionadas pelo contexto (mesmo sem ser sinônimos) ficam próximas
- **Direções vetoriais:** relações recorrentes se tornam vetores direcionais (ex: analogias)
- **Contexto:** o modelo lembra apenas os tokens presentes no prompt atual
- **Atenção:** Self-Attention permite decidir **quem influencia quem** em cada passo

---

## ⚠️ Alucinações e Limitações

- LLMs **não sabem o que é verdade ou mentira**
- Apenas geram **tokens prováveis dado o contexto**
- Podem criar informações falsas que parecem convincentes

Para reduzir isso:

- Forneça **contexto completo**
- Permita respostas do tipo `"não sei"`
- Evite prompts que exijam certeza absoluta

---

# 🧩 8️⃣ Quando usar LLMs?

- Geração de texto (emails, artigos, resumos)
- Respostas automáticas e chatbots
- Tradução de idiomas
- Revisão e criação de código
- Assistência pessoal ou educacional
- Qualquer tarefa que envolva **linguagem natural complexa**

---

# 📌 9️⃣ Resumo Final

Um LLM:

1. **Tokeniza** o texto
2. Representa tokens como **embeddings**
3. Processa com **Transformer e Self-Attention**
4. Gera tokens usando **probabilidades** e **parâmetros de geração**
5. Captura **similaridade, associação e relações semânticas**
6. Permite **operações vetoriais e analogias** (ex: verão – quente + frio ≈ inverno // rei – homem + mulher = rainha)
7. Gera o texto **token por token (sampling)**

> Ele **não entende como nós entendemos**, mas consegue gerar texto coerente e útil baseado em padrões aprendidos.
