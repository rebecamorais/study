import { AIService } from './services/aiService.js';
import { TranslationService } from './services/translationService.js';
import { View } from './views/view.js';
import { FormController } from './controllers/formController.js';

(async function main() {
    // 1. Registro do botão de problemas (sempre disponível)
    const forceButton = document.getElementById('force-download-btn');
    if (forceButton) {
        forceButton.addEventListener('click', async () => {
             console.log("Iniciando força-tarefa de download...");
             try {
                /*
                    * Como as APIs de IA nativas do Chrome (Gemini Nano) estão em fase 
                    * experimental (Canary v147), o namespace e o contrato mudam frequentemente.
                    *   1. window.ai.languageModel: Padrão atual para acesso ao Gemini Nano (v147+).
                    *   2. window.ai.assistant: Namespace anterior (legado de builds v145-).
                    *   3. window.LanguageModel: Fallback para injeções globais diretas.
                    * Daí o que tentamos fazer é: 
                    * Tentamos instanciar o Translator simultaneamente porque ambos compartilham o 
                    * motor 'Optimization Guide'. Acordar um costuma destravar o download do outro.
                    *  https://github.com/webmachinelearning/prompt-api aqui tem mais info
                */


                 const AI = window.ai?.languageModel || window.ai?.assistant || window.LanguageModel;
                 if (AI) {
                     console.log("Tentando acordar Gemini Nano...");
                     AI.create({ expectedOutputLanguages: ["en"] }).catch(() => { console.log('erro AI.create({ expectedOutputLanguages: ["en"] })')});
                 }
 
                 const TranslatorAPI = window.translation || window.ai?.translation || window.ai?.translator || window.Translator;
                 if (TranslatorAPI) {
                     console.log("Tentando acordar Translator...");
                     TranslatorAPI.create({ sourceLanguage: 'en', targetLanguage: 'pt' }).catch(() => {});
                 }
 
                 alert("Comandos enviados! Verifique chrome://components.\n\nSe 'Optimization Guide' ou 'Translation' ainda não estiverem ativos, REINICIE o Chrome.");
             } catch (e) {
                 alert("Erro: " + e.message);
             }
        });
    }

    // Initialize services and view
    const aiService = new AIService();
    const translationService = new TranslationService();
    const view = new View();
    
    // Set current year
    view.setYear();

    // Check requirements
    const errors = await aiService.checkRequirements();
    if (errors) {
        view.showError(errors);
        return;
    }

    // Initialize translation services
    try {
        await translationService.initialize();
    } catch (error) {
        console.error('Error initializing translation (non-blocking):', error);
    }

    // Get and initialize AI parameters
    const params = await aiService.getParams();
    view.initializeParameters(params);

    // Initialize controller and setup event listeners
    const controller = new FormController(aiService, translationService, view);
    controller.setupEventListeners();

    console.log('Application initialized successfully');
})();
