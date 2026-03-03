# 🌱 Algoritmos Genéticos

Algoritmos Genéticos (AGs) operam sobre uma população de indivíduos (potenciais soluções), aplicando operadores que imitam a genética real para “evoluir” a melhor resposta ao longo de várias gerações.

### EM OUTRAS PALAVRAS:

Em vez de tentar encontrar a melhor solução direto, calculando tudo:

- Cria várias soluções aleatórias
- Testa quais são melhores
- Deixa as melhores “terem filhos”
- Mistura características delas
- Faz pequenas mudanças aleatórias
- Repete isso várias vezes

👉 No fim, as soluções vão ficando cada vez melhores.

**Sobrevive quem resolve melhor o problema.**

---

# 🧬 1. Conceitos Básicos

Alguns termos mais ligados a biologia são os termos oficiais da literatura, não necessáriamente a gente vai ter isso no código do dia a dia.

### 🔹 Indivíduo (ou Cromossomo)

É **uma solução candidata**. Geralmente vai ser uma lista/array

Tecnicamente: é representado como uma sequência de dados (ex: binário ou números).

> É uma possível resposta para o problema.

`individuo = [1, 1, 0, 0, 1]`

Isso pode representar um número (ex: 25).

---

### 🔹 Gene - Locus - Alelo

É a menor unidade de informação do cromossomo. Um elemento do array.

```
individuo = [1, 1, 0, 0, 1]

Posição:   0 1 2 3 4
Valor:     1 1 0 0 1
```

No exemplo: `individuo[2] seria o 0`

_Gene_ = elemento da lista

_Locus_ = índice

_Alelo_ = valor retornado

### 🔹 População

É o conjunto de indivíduos de uma geração. Vai ser uma lista de listas provavelmente.

Exemplo:

```
populacao = [
    [1,1,0,0,1],
    [0,0,0,1,1],
    [1,0,1,1,0]
]
```

Esses quatro cromossomos (arrays) formam uma população.

---

### 🔹 Função de Fitness (Aptidão)

É a função que diz **o quão boa é a solução**.

Ela dá uma nota.

Exemplo:

Se queremos maximizar: `f(x) = x²`

```
def fitness(individuo):
    return individuo * individuo
```

Se o indivíduo representa 5:

- Fitness = 25

Se representa 10:

- Fitness = 100

Quanto maior o fitness, melhor a solução.

É literalmente o critério de sobrevivência.

---

# 🔁 2. O Ciclo Evolutivo (Passo a Passo)

O AG funciona em ciclos repetitivos chamados gerações.

Suponha que o que queremos com o nosso AG é: _Maximizar o valor do número representado em binário._

---

## 1️⃣ Inicialização

Criamos uma população aleatória.

Exemplo:

```
01101
10010
00011
11100
```

Nada é especial ainda — é tudo tentativa inicial.

---

## 2️⃣ Avaliação

Calculamos o **fitness** de cada indivíduo.

No nosso exemplo específico:

- O indivíduo está representado em binário.
- Convertimos ele para decimal.
- Como queremos maximizar o valor do número, o fitness será esse valor decimal.

`fitness(indivíduo) = valor_decimal(indivíduo)`

Agora sabemos quem é melhor.
👉 O indivíduo que o fitness é o maior de todos.

```
    Representação (genótipo)
        ↓
    Interpretação
        ↓
    Função de fitness
        ↓
    Comparação entre indivíduos
```

---

## 3️⃣ Seleção

Escolhemos indivíduos para serem pais.

Regra geral:

> Quem tem fitness maior tem mais chance de ser escolhido.

_Importante_:
Não significa que só os melhores sobrevivem — mas eles têm mais probabilidade.

**🔹 Como funciona na prática**

- Não significa que só o melhor sobrevive.
- Cada indivíduo tem uma probabilidade proporcional ao fitness de ser selecionado.
- Podemos selecionar mais de um para gerar vários filhos.
- Da pra calcular uma probabilidade usando _(fitness do individuo / soma total dos fitness)_ e usar essa % pra decidir quais vamos selecionar.

Exemplo:

- Queremos gerar 4 filhos.
- Selecionamos 2 pais → aplicamos crossover → geramos 2 filhos
- Selecionamos mais 2 pais → geramos mais 2 filhos
- Assim completamos a próxima geração.

---

## 4️⃣ Crossover (Recombinação)

É o principal operador de recombinação e intensificação da busca. No código, vai ser uma função que combina os arrays (indivíduos) que tiveram bom fitness.

Do jeito que eu entendo:

> Pegamos duas soluções boas e misturamos partes delas.

Exemplo (corte no meio):

Pai A: `11011` (fitness 28)

Pai B: `00100` (fitness 18)

Cortando:

```
Pai 1: 111 | 00
Pai 2: 100 | 10
```

Ao misturar as partes, podemos gerar 1 ou 2 filhos:

**Filho 1**: `11110  (metade de pai1 + metade de pai2)`

**Filho 2**: `10000  (metade de pai2 + metade de pai1)`

- O filho pode ter fitness maior ou menor que os pais.
- Nós só estamos experimentando combinações.

---

## 5️⃣ Mutação

Alteração aleatória em genes específicos de um indivíduo.

Serve para:

- **Manter diversidade**: evita que todos os indivíduos fiquem iguais muito rápido
- **Evitar mínimos locais**: ajuda a população a escapar de soluções que são boas, mas não ótimas
- **Explorar novas possibilidades**: permite que surjam combinações que não existiam antes

🔹 **Como funciona no exemplo**

Suponha que temos um filho gerado pelo crossover:

- Escolhemos aleatoriamente um gene para alterar.
- Não é obrigatório mutar todos os indivíduos
- Normalmente só alguns são escolhidos aleatoriamente
- Cada gene de cada indivíduo tem uma pequena chance de mutar (ex.: 1–5%) para não transformar o algoritmo em pura aleatoriedade
- Mutação é aleatória, não escolhe “melhor bit”

```
Antes:  11000
Depois: 11100
```
---

## 6️⃣ Substituição

A nova geração substitui a antiga (mesmo tamanho da anterior).

Aqui pode existir **Elitismo**:

> Garantir que o melhor indivíduo da geração atual não seja perdido.

Ou seja:
Mesmo que algo dê errado, o melhor continua.

🔹 Por que elitismo é importante

- Evita perder soluções muito boas por acaso
- Acelera convergência para soluções ótimas
- Mas **cuidado**: elitismo demais pode reduzir diversidade e causar convergência prematura

---

# 🧠 Conceitos de Representação

### 🔹 Genótipo e Fenótipo

* **Genótipo:** codificação interna do indivíduo (os “genes brutos”)
* **Fenótipo:** a interpretação real do genótipo, o que ele significa no mundo real

Exemplo:

```
Genótipo: 11001                         (sequência de bits ou vetor)
Fenótipo: 25  (valor decimal)           (solução que realmente queremos avaliar)
```

---

### 🔎 Exploration (Exploração)

* Buscar novas regiões do espaço de soluções
* Evitar ficar preso em soluções locais
* Quem ajuda: **mutação**

> Se explorar demais → vira pura aleatoriedade (nenhuma solução consistente)

---

### 🎯 Exploitation (Aproveitamento)

* Aproveitar as soluções boas que já encontramos
* Quem ajuda: **seleção + crossover**

> Se aproveitar demais → a população fica parecida rápido → **convergência prematura**

---

# ⚠️ Convergência Prematura

* População perde diversidade rápido
* Todos os indivíduos ficam muito parecidos
* Resultado: o algoritmo para em um **mínimo local** (solução boa, mas não a melhor possível)

> Por isso a mutação é essencial: ela mantém diversidade e ajuda a escapar de mínimos locais.

---

# 🧩 5️⃣ Quando usar Algoritmos Genéticos?

Use AG quando:

* O problema tem **muitas combinações possíveis**
* Não dá para testar todas (busca muito grande)
* Não existe **fórmula direta** para a solução ótima

Exemplos:

* Otimização combinatória (Problema do Caixeiro Viajante)
* Ajuste de hiperparâmetros em modelos
* Neuroevolução (design de redes neurais)
* Qualquer problema com espaço de busca **muito grande**

---

# 📌 Resumo Final

Um Algoritmo Genético:

1. Trabalha com **várias soluções ao mesmo tempo** (população)
2. Avalia quais são melhores (**fitness**)
3. Mistura as melhores (**crossover**)
4. Introduz pequenas variações (**mutação**)
5. Repetem o processo por várias gerações

Com o tempo, as soluções **evoluem** e ficam cada vez melhores.

> Ele **não garante a solução perfeita**, mas é eficiente para encontrar soluções muito boas em problemas difíceis.