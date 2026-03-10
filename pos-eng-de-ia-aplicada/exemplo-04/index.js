const AI = window.ai?.languageModel || window.ai?.assistant || window.LanguageModel;

const aiContext = {
    session: null,
    abortController: null,
    isGenerating: false,
};

const elements = {
    temperature: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temp-value'),
    topKValue: document.getElementById('topk-value'),
    topK: document.getElementById('topK'),
    form: document.getElementById('question-form'),
    questionInput: document.getElementById('question'),
    output: document.getElementById('output'),
    button: document.getElementById('ask-button'),
    year: document.getElementById('year'),
}

async function setupEventListeners() {

    // Update display values for range inputs
    elements.temperature.addEventListener('input', (e) => {
        elements.temperatureValue.textContent = e.target.value;
    });

    elements.topK.addEventListener('input', (e) => {
        elements.topKValue.textContent = e.target.value;
    });

    elements.form.addEventListener('submit', async function (event) {
        event.preventDefault();

        if (aiContext.isGenerating) {
            toggleSendOrStopButton(false)
            return;
        }

        onSubmitQuestion();
    });
}

async function onSubmitQuestion() {
    const questionInput = elements.questionInput;
    const output = elements.output;
    const question = questionInput.value;

    if (!question.trim()) {
        return;
    }

    // Get parameters from form
    const temperature = parseFloat(elements.temperature.value);
    const topK = parseInt(elements.topK.value);
    console.log('Using parameters:', { temperature, topK });

    // Change button to stop mode
    toggleSendOrStopButton(true)

    output.textContent = 'Processing your question...';
    const aiResponseChunks = await askAI(question, temperature, topK);
    output.textContent = '';

    for await (const chunk of aiResponseChunks) {
        if (aiContext.abortController.signal.aborted) {
            break;
        }
        console.log('Received chunk:', chunk);
        output.textContent += chunk;
    }

   toggleSendOrStopButton(false);
}

function toggleSendOrStopButton(isGenerating) {
    if (isGenerating) {
        // Switch to stop mode
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Parar';
        elements.button.classList.add('stop-button');
    } else {
        // Switch to send mode
        aiContext.abortController?.abort();
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Enviar';
        elements.button.classList.remove('stop-button');
    }
}
async function* askAI(question, temperature, topK) {
    aiContext.abortController?.abort();
    aiContext.abortController = new AbortController();

    if (aiContext.session) {
        try {
            aiContext.session.destroy();
        } catch (e) {}
    }

    console.log('Criando sessão com Gemini Nano...');
    
    // systemPrompt é uma string, não um array ? 
    const sessionConfig = {
        temperature: temperature,
        topK: topK,
        systemPrompt: "Você é um assistente de IA que responde de forma clara e objetiva em português. Responda sempre em formato de texto ao invés de markdown."
    };

    try {
        const session = await AI.create(sessionConfig);
        aiContext.session = session;

        console.log('Sessão criada. Enviando pergunta:', question);
        
        // Nas versões novas, promptStreaming recebe uma string direta
        const responseStream = await session.promptStreaming(question, {
            signal: aiContext.abortController.signal,
        });

        for await (const chunk of responseStream) {
            if (aiContext.abortController.signal.aborted) break;
            yield chunk;
        }
    } catch (e) {
        console.error('Erro na sessão de IA:', e);
        yield `Erro ao processar: ${e.message}. Verifique se o modelo terminou de baixar em chrome://components`;
    }
}

async function checkRequirements() {
    const errors = [];
    const returnResults = () => errors.length ? errors : null;

    // @ts-ignore
    const isChrome = !!window.chrome;
    if (!isChrome)
        errors.push("⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente).");
    if (!AI) {
        errors.push("⚠️ As APIs nativas de IA não estão ativas.");
        errors.push("Ative a seguinte flag em chrome://flags/:");
        errors.push("- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)");
        errors.push("Depois reinicie o Chrome e tente novamente.");
        return returnResults();
    }

    const availability = await AI.availability({ 
        expectedInputLanguages: ["en"],
        expectedOutputLanguages: ["en"] 
    });
    console.log('Language Model Availability:', availability);
    if (availability === 'available') {
        return returnResults();
    }

    if (availability === 'unavailable') {
        errors.push(`⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA.`);
    }

    if (availability === 'downloading') {
        errors.push(`⚠️ O modelo de linguagem de IA está sendo baixado.`);
        errors.push(`Acompanhe o progresso em chrome://components procurando por "Optimization Guide On Device Model".`);
        errors.push(`Por favor, aguarde alguns minutos e tente novamente.`);
    }

    if (availability === 'downloadable') {
        errors.push(`⚠️ O modelo de linguagem de IA precisa ser baixado, baixando agora... (acompanhe o progresso no terminal do chrome)`);
        try {
            const session = await AI.create({
                expectedInputLanguages: ["en"],
                expectedOutputLanguages: ["en"],
                monitor(m) {
                    m.addEventListener('downloadprogress', (e) => {
                        const percent = ((e.loaded / e.total) * 100).toFixed(0);
                        console.log(`Downloaded ${percent}%`);
                    });
                }
            });
            await session.prompt('Olá');
            session.destroy();

            // Re-check availability after download
            const newAvailability = await AI.availability({ 
                expectedInputLanguages: ["en"],
                expectedOutputLanguages: ["en"]
            });
            if (newAvailability === 'available') {
                return null; // Download successful
            }
        } catch (error) {
            console.error('Error downloading model:', error);
            errors.push(`⚠️ Erro ao baixar o modelo: ${error.message}`);
        }
    }

    return returnResults();

}

// 1. Forçar registro do botão de problemas o mais rápido possível
(function initTroubleshooting() {
    const forceButton = document.getElementById('force-download-btn');
    if (!forceButton) {
        console.error("Botão #force-download-btn não encontrado no HTML!");
        return;
    }

    forceButton.addEventListener('click', async () => {
        console.log("Iniciando força-tarefa de download...");
        try {
            // Método 1: Prompt API (Gemini Nano)
            const promptAPI = window.ai?.languageModel || window.ai?.assistant || window.LanguageModel;
            if (promptAPI) {
                console.log("Tentando acordar Gemini Nano...");
                // @ts-ignore
                promptAPI.create({ expectedOutputLanguages: ["en"] }).catch(() => {});
            }

            // Método 2: Translation API
            // @ts-ignore
            const translator = window.translation || window.ai?.translation || window.ai?.translator;
            if (translator) {
                console.log("Tentando acordar Translator...");
                // @ts-ignore
                const can = await (translator.canTranslate ? 
                    translator.canTranslate({ sourceLanguage: 'en', targetLanguage: 'pt' }) : 
                    translator.availability({ sourceLanguage: 'en', targetLanguage: 'pt' }));
                
                if (can !== 'no') {
                    // @ts-ignore
                    translator.create({ sourceLanguage: 'en', targetLanguage: 'pt' }).catch(() => {});
                }
            }

            alert("Comandos de ativação enviados!\n\n1. Verifique chrome://components\n2. Se 'Optimization Guide' ainda não aparecer, REINICIE o Chrome (feche tudo).\n3. Às vezes o Chrome demora 1-2 minutos para listar o componente após o comando.");

        } catch (e) {
            console.error("Erro no clique do botão:", e);
            alert("Erro: " + e.message);
        }
    });
})();


(async function main() {
    // Proteção contra elementos ausentes
    if (elements.year) elements.year.textContent = new Date().getFullYear();

    const reqErrors = await checkRequirements();
    if (reqErrors) {
        if (elements.output) elements.output.innerHTML = reqErrors.join('<br/>');
        if (elements.button) elements.button.disabled = true;
        return;
    }

    try {
        console.log('Checking AI capabilities with object:', AI);
         console.log('AI.capabilities', Object.keys(AI))
        // Versão ultra-robusta: verifica se é função antes de chamar
        let capacities = {};
        if (typeof AI.capabilities === 'function') {
            console.log('AI.capabilities')
            capacities = await AI.capabilities();
        } else if (typeof AI.params === 'function') {
            console.log('AI.params')
            capacities = await AI.params();
        } else if (AI.params && typeof AI.params === 'object') {
            console.log('AI. objeto')
            capacities = AI.params;
        }
        
        console.log('Language Model Capabilities:', capacities); // N sei mas ta sempre obejto vazio ??
        
        const defaultTopK = capacities.defaultTopK || 3;
        const maxTopK = capacities.maxTopK || 128;
        const defaultTemp = capacities.defaultTemperature || 1;
        const maxTemp = capacities.maxTemperature || 2;

        if (elements.topK) {
            elements.topK.max = maxTopK;
            elements.topK.min = 1;
            elements.topK.value = defaultTopK;
            elements.topKValue.textContent = defaultTopK;
        }

        if (elements.temperature) {
            elements.temperatureValue.textContent = defaultTemp;
            elements.temperature.max = maxTemp;
            elements.temperature.min = 0;
            elements.temperature.value = defaultTemp;
        }
        console.log('UI inicializada com sucesso.');
        return setupEventListeners();
    } catch (e) {
        console.error("Erro ao carregar parâmetros da IA, usando padrões seguros:", e);
        // Fallback total para a página não travar
        if (elements.topK) elements.topK.value = 3;
        if (elements.temperature) elements.temperature.value = 1;
        return setupEventListeners();
    }
})();

