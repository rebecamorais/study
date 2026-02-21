import tf, { model } from '@tensorflow/tfjs-node';

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [7],        // 7 posicoes: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
        units: 80,              // 80 neuronios pq a gnt tem pouca base de treino
        activation: 'relu'      // filtro de ativação - (Rectified Linear Unit) é uma função de ativação não linear que retorna o valor de entrada se for positivo, e 0 caso contrário 
    }));

    model.add(tf.layers.dense({
        units: 3,                       // Camada de saida com 3 neuronios pq a gnt tem 3 categorias (premium, medium, basic)
        activation: 'softmax'           // softmax para classificacao
    }));

    model.compile({
        optimizer: 'adam',              // otimizador - Adaptive Moment Estimation - ajusta os pesos  (aprende com o historico de erros e acertos)
        loss: 'categoricalCrossentropy', // compara a probabilidade prevista (calculo q o modelo achou) com a probabilidade real (resposta certa)
        metrics: ['accuracy']           // para avaliar o desempenho do modelo - se a taxa de erro for mto alta, o modelo não aprendeu nada etc
        // A resposta certa sempre vai ser uma entre as categorias possiveis. Ex: [1, 0, 0] ou [0, 1, 0] ou [0, 0, 1]
        // O modelo vai tentar prever a resposta certa, comparando com a resposta certa, ele vai ajustar os pesos
    });

    // Treinamento do modelo
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,    // 0 = não mostra nada, 1 = mostra o progresso, 2 = mostra o progresso detalhado
            epochs: 100,   // qtd de vezes q roda na lista de input 
            shuffle: true, // embaralha os dados a cada época, pra não viciar o modelo em uma ordem específica de entrada dos dados
            callbacks: {
                // onEpochEnd: (epoch, logs) => {
                //     console.log(`Epoch ${epoch + 1}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
                // }
            }
        });
    return model;
}

async function predict(model, inputList) {
    console.log(inputList);

    const tfInput = tf.tensor2d(inputList); // Monta o input pro modelo

    const prediction = model.predict(tfInput);

    const predArray = await prediction.array();
    return predArray[0].map((prob, index) => {
        return {
            index,
            label: labelsNomes[index],
            prob,
        }
    });
    // const predictedClass = prediction.argMax(1).dataSync()[0];
    // return labelsNomes[predictedClass];
}

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const myModel = await trainModel(inputXs, outputYs);

const pRebeca = {
    nome: "Rebeca",
    idade: 29,
    cor: "verde",
    localizacao: "São Paulo"
}

const pZe = {
    nome: 'zé',
    idade: 28,
    cor: 'verde',
    localizacao: 'Curitiba'
}

const normalizarPessoa = (pessoa) => {
    return [
        pessoa.idade / 100, // Normaliza a idade dividindo por 100 ? 
        pessoa.cor === "azul" ? 1 : 0,
        pessoa.cor === "vermelho" ? 1 : 0,
        pessoa.cor === "verde" ? 1 : 0,
        pessoa.localizacao === "São Paulo" ? 1 : 0,
        pessoa.localizacao === "Rio" ? 1 : 0,
        pessoa.localizacao === "Curitiba" ? 1 : 0
    ]
}

// Normalizando os dados da pessoa para que o modelo possa entender aaa
const pRebecaNormalizada = normalizarPessoa(pRebeca);
const pZeNormalizado = [
    0.2,
    0,
    0,
    1,
    0,
    0,
    1
];

const listaDeInputs = [pZeNormalizado];

const predictions = await predict(myModel, listaDeInputs);

const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(({ label, prob }) => `${label}: ${(prob * 100).toFixed(2)}%`)
    .join('\n');
console.log(results);