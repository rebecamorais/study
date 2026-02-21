import tf from '@tensorflow/tfjs-node';

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

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

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const models = trainModel(inputXs, outputYs);