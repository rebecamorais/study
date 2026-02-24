import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let _globalCtx = {};
let _model = null

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};


// 🔢 Normalize continuous values (price, age) to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    // Computar a média de idade dos comprados por produto
    // (ajuda a personalizar)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    const productAvgAge = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, avg] 
            // Fiz isso pq tava confusa se ja tinha normalizado ou nao o vetor 
            // e tava na duvida se normalizaria 2x por causa do BuildVector
            // Desse jeito eu só salvo a média (nao normalizada ainda)
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAge,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimentions: 2 + categories.length + colors.length
    }
}

const oneHotWeighted = (index, length, weight) =>  // Isso pra categoria/cor virar um numero entre 0 e 1 
    tf.oneHot(index, length).cast('float32').mul(weight)

function buildVector(context, { price, age, category, color } = {}) {
    // Se o valor existir, normaliza e aplica o peso | se não preenche com zero
    
    // Preço
    const priceTensor = price !== undefined ? 
        tf.tensor1d([normalize(price, context.minPrice, context.maxPrice) * WEIGHTS.price]) : 
        tf.zeros([1]);

    // Idade (seja do usuário ou a média do produto)
    // Quando for usuario eu passo a idade dele
    // Quando for produto eu passo a média de idade dos usuarios que compraram ele
    
    const ageValue = age !== undefined ? age : (context.minAge + context.maxAge) / 2;
    const ageTensor = tf.tensor1d([
        normalize(ageValue, context.minAge, context.maxAge) * WEIGHTS.age
    ]);

    // Categoria
    const categoryTensor = category !== undefined ?
        oneHotWeighted(context.categoriesIndex[category], context.numCategories, WEIGHTS.category) :
        tf.zeros([context.numCategories]);

    // Cor
    const colorTensor = color !== undefined ?
        oneHotWeighted(context.colorsIndex[color], context.numColors, WEIGHTS.color) :
        tf.zeros([context.numColors]);

    return tf.concat1d([priceTensor, ageTensor, categoryTensor, colorTensor]);
}

function encodeProduct(product, context) {
    return buildVector(context, {
        price: product.price,
        age: context.productAvgAge[product.name], // Passando valor bruto
        category: product.category,
        color: product.color
    });
}


function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(product => encodeProduct(product, context))
        ).mean(0)
        .reshape([1, context.dimentions])
    }

    // Agora é automático! Se novos campos surgirem, o buildVector cuida de por zeros
    return buildVector(context, { age: user.age })
        .reshape([1, context.dimentions]);
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
    .filter(u => u.purchases.length)
    .forEach(user => {
        const userVector = encodeUser(user, context).dataSync();
        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync();
            const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;

            inputs.push([...userVector, ...productVector])
            labels.push(label)
        })
    })
    return { 
        xs: tf.tensor2d(inputs), 
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2               // pq o tamanho vai ser = userVector + productVector
    }
}

async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential();
    
    // Arquitetura em Funil pra extrair melhor caracteristicas dos dados
    model.add(  
        // Aqui primeiro vai tentar pegar o maximode relacoes possiveis entre os dados brutos
        tf.layers.dense({
            inputShape: [trainData.inputDimention], // inputDimention: context.dimentions * 2
            units: 128,
            activation: 'relu'  // pra eu nao esquecer q o relu é pra tipo "se o usuário tem 20 anos(jovem?) E o produto é caro E azul, a chance de compra é X"
        })
    )

    model.add(
        // depois de ja ter treinado com os dados brutos, tenta fazer tipo um resumo(?) das relacoes possiveis
        tf.layers.dense({
            // inputShape: [trainData.inputDimention],  só precisa na primeira camada, nas outras ele pega automaticamente
            units: 64,
            activation: 'relu'
        })
    )

    model.add(
        // em cima do resumo, tenta fazer de novo a partir disso das relacoes possiveis
        tf.layers.dense({
            // inputShape: [trainData.inputDimention],
            units: 32,
            activation: 'relu'
        })
    )

    model.add(tf.layers.dense({
        units: 1,                 // Quero q o resultado final seja só a probabilidade de compra
        activation: 'sigmoid'     // pra garantir q o resultado seja entre: baixa prob(0) e alta prob(1)   
    }))

    model.compile({
        optimizer: tf.train.adam(0.01), // Otimizador: Adam (ajusta os pesos automaticamente)
        loss: 'binaryCrossentropy',      // Função de perda: Binary Crossentropy (mede o erro entre a previsão e o valor real)
        metrics: ['accuracy']            // Métricas: Accuracy (mede a precisão do modelo)
    })

    await model.fit(
        trainData.xs,         // Entradas - as perguntas
         trainData.ys,        // Os resultados - as respostas certas - pra combuinacao xs o resultado foi 1 se comprou e 0 se nao comprou
          {
        epochs: 100, //  quantas vezes o modelo vai ver os dados - se for demais o modelo pode decorar as respotas, se for pouco pode nao parender os padroes
        batchSize: 32, //  quantos dados o modelo vai ver de cada vez - pra num ler tudao de uma so vez
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                // loss: "quão errado o modelo está" - quanto menor melhor
                // accuracy: "quão certo o modelo está" - quanto maior melhor
                postMessage({ type: workerEvents.trainingLog, epoch, loss: logs.loss, accuracy: logs.acc });
            }
        }
    })

    return model
}

async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    const products = await (await fetch('/data/products.json')).json() 

    const context = makeContext(products, users) 
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync(),
        }
    }) 

    _globalCtx = context

    const trainData = createTrainingData(context);

    _model = await configureNeuralNetAndTrain(trainData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend({ user }) {
    if(!_model) return;

    const context = _globalCtx
    const userTensor = encodeUser(user, context).dataSync()


    const inputs = context.productVectors.map(({ vector }) => { // Comparando o perfil da pessoa userVector com o produto
        return [...userTensor, ...vector]
    });

    const inputTensor = tf.tensor2d(inputs)

    const predictions = _model.predict(inputTensor)

    const scores = predictions.dataSync()
    
    
    const recommendations = context.productVectors.map((product, index) => {
        return {
            ...product.meta,
            name: product.name,
            score: scores[index]
        }
    })

    const sortedRecomendations = recommendations.sort((a,b) => b.score - a.score)

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecomendations
    });

}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
