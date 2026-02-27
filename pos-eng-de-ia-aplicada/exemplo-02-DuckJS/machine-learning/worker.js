importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENSIONS = 640
const CLASS_THRESHOLD = 0.4

let _labels = []
let _model = null

async function loadModelAndLabels() {
    await tf.ready()

    _labels = await (await fetch(LABELS_PATH)).json();
    _model = await tf.loadGraphModel(MODEL_PATH);

    // (warmup) Aquecimento - faz uns alongamentos ai
    // Isso precisa pq o modelo nao ta totalmente compildo na hora que ele é carregado, 
    // então a primeira inferência é sempre mais lenta
    // se num fizer isso, a primeira inferência da posição dos patinhos vai ser bem mais lenta
    // Ou seja o usuário pode ter a impressão de que o clique/frame tá travado 
    const dummyInput = tf.ones(_model.inputs[0].shape)
    await _model.executeAsync(dummyInput)
     // Libera a memória da GPU manualmente, já que o JS não limpa tensores automaticamente
     // Se não fizer isso, a memória da GPU vai acumular e o navegador vai travar
     // pq o garbage collector do JS não consegue limpar a memória da GPU
    tf.dispose(dummyInput)       
    postMessage({ type: 'model-loaded' })
}

function preprocessImage(input) {
    return tf.tidy(() => { // versão inteligente do tf.dispose: evita que a gente tenha que limpar cada variável da GPU na mão  - Tudo que nasceu aqui dentro e não foi retornado, pode jogar fora
        
        const tensor = tf.browser.fromPixels(input) //  converte uma imagem num tensor (aquela matriz de numeros)
        // Aqui estamos fazendo a mesma coisa que no exemplo-01 > pegando os dados das pessoas e transformando em um negocio tipo assim: [[1,0,0,0.5]]
        // Só que ao invés de a gente fzer os calculos na mao o tensorflow ja faz isso pra gnt com o browser.fromPixels 
        // O pulo do gato haha é que esse é um tensor 3D [H,W,3] ou em outras plavras height width e 3 canais de cor (RGB)
        
        const resizedTensor = tf.image.resizeBilinear(tensor, [INPUT_MODEL_DIMENSIONS, INPUT_MODEL_DIMENSIONS]) // O modelo espera uma imagem 2D por isso a gente precisa fazer o resize

        const normalizedTensor = resizedTensor.div(255) // O modelo espera valores entre 0 e 1, por isso a gente precisa fazer o normalize
        // PQ 255 ? Pq estamos falando de RGB onde 
        // 0 = preto             continua sendo zero   (0 / 255 = 0)
        // 255 = branco          vira 1     (255 / 255 = 1)
        // todo o resto fica entre 0 e 1    (ex: 127 vira ~0.5).

        const batchedTensor = normalizedTensor.expandDims(0) // O modelo espera um tensor 4D [batch, height, width, channels], por isso a gente precisa fazer o expandDims
        //  [batch,      height, width,    channels]
        //  [Quantidade, Altura, Largura, Canais] 

        return batchedTensor; // Como só retornei o batchedTensor, ele continuará existindo, mas o tidy vai apagar todo o resto automaticamente
        // Agora vem a magica do tidy, todas as coisas q a gente criou aqui vao ser "apagadas" pq n precisamos mais delas
        // A gente precisaria fazer tf.dispose pra cada uma das variaveis se n fosse o tidy
    
    })
}

async function runInference(tensorImage) {
    const output = await _model.executeAsync(tensorImage)
    tf.dispose(tensorImage) // já transfor meu tensorImage numa saida e só vou usar ela daqui por diante entao nao preciso mais dele logo = joga fora
    const [ boxes, scores, classes] = output.slice(0, 3)
    // o output é um array de uns objetos - eles nao estao em formato de array pq eles estao como um objeto do tipo tf.Tensor (onde os dados moram)
    // boxes: sao as coordenadas de onde a IA acha que os objetos estao tipo um retangulo
    // scores: sao as confiancas de que aqueles objetos sao realmente o que a IA acha que sao
    // classes: sao as classes dos objetos

    const [ boxesData, scoresData, classesData] = await Promise.all([  // AGOOOORA a gnt vai transformar isso em array
        boxes.data(),
        scores.data(),
        classes.data(),
    ])

    output.forEach(t => tf.dispose(t)) // Jogando fora o [ boxes, scores, classes] pq nao precisamos mais deles
    // pq nao usar tf.tidy() ?? pq ele não aceita funções assíncronas (async/await) entao na hora de um await ele poderia entender q ja acabou e apagar os tensores
    // acaba que o tf.dispose fica sendo obrigatório qnd tiver um await no meio
    return {
        boxes: boxesData,
        scores: scoresData,
        classes: classesData
    }
}

// Função Geradora (*): 
// Em vez de entregar todos os patinhos de uma vez num pacote (array), ela entrega um por um conforme a gente pede, "pausando" a execução entre cada entrega.
// Isso é util pq a gente pode processar os resultados sem travar o navegador
function* processPrediction(inferenceResults, width, height) {
    const { boxes, scores, classes} = inferenceResults;

    for (let index = 0; index < scores.length; index++) {
        const label = _labels[classes[index]];
        if(label !== 'kite') continue       // Se num é kite ja vou ignorar

        if(scores[index] < CLASS_THRESHOLD) continue // Se só tem 40% de certeza que é um kite, vou ignorar 

        // Se o index for 0, ele vai pegar do 0 ao 4, se for 1, vai pegar do 4 ao 8, e assim por diante
        let [x_min, y_min, x_max, y_max] = boxes.slice(index*4, (index+1)*4)
    
        // Os valores da IA vêm entre 0 e 1, então fazemos a "Denormalização" (regra de 3)
        // Multiplicamos X pela largura (width) e Y pela altura (height)
        x_min *= width         
        x_max *= width         
        y_min *= height
        y_max *= height

        // Esses cálculos servem para achar o centro do pato e saber onde atirar
        // y_min  |---------------------------------------|
        // (Topo) |                                       |
        //        |                   *                   |  <-- (centerX, centerY)
        // y_max  |                                       |
        // (Base) |---------------------------------------|
        //      x_min (Esquerda)                 x_max (Direita)
        //       ___________ Largura (x_max - x_min) ___________


        const boxWidth = x_max - x_min
        const boxHeight = y_max - y_min

        // O centro é o ponto inicial + metade da largura
        const centerX = x_min + (boxWidth / 2)            // teeeeecnicamente a média aritmetica daria a msma coisa ? 
        const centerY = y_min + (boxHeight / 2)           
        

       yield  {
            x: centerX,
            y: centerY,
            score: (scores[index] *100).toFixed(2)
       }
    }
    
}

loadModelAndLabels();

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if (!_model) return

    const tensorImage = preprocessImage(data.image) // Transforma a imagem em tensores

    const inferenceResults = await runInference(tensorImage); // o modelo analisa os pixels e nos diz onde estão os objetos e qual a confiança de cada um

    const { width, height } = data.image;
    
    for (const prediction of processPrediction(inferenceResults, width, height)) {
        postMessage({
            type: 'prediction',
            ...prediction,
        });
    }
};

console.log('🧠 YOLOv5n Web Worker initialized');
