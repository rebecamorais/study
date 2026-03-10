export class AIService {
    constructor() {
        this.session = null;
        this.abortController = null;
    }

    async checkRequirements() {
        const errors = [];

        // @ts-ignore
        const isChrome = !!window.chrome;
        if (!isChrome) {
            errors.push("⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente).");
        }

        if (!('LanguageModel' in self)) {
            errors.push("⚠️ As APIs nativas de IA não estão ativas.");
            errors.push("Ative a seguinte flag em chrome://flags/:");
            errors.push("- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)");
            errors.push("Depois reinicie o Chrome e tente novamente.");
            return errors;
        }

        // Check Translator availability
        if ('Translator' in self) {
            const translatorAvailability = await Translator.availability({
                sourceLanguage: 'en',
                targetLanguage: 'pt'
            });
            console.log('Translator Availability:', translatorAvailability);

            if (translatorAvailability === 'no') {
                errors.push("⚠️ Tradução de inglês para português não está disponível.");
            }
        } else {
            errors.push("⚠️ A API de Tradução não está ativa.");
            errors.push("Ative a seguinte flag em chrome://flags/:");
            errors.push("- Translation API (chrome://flags/#translation-api)");
        }

        // Check Language Detection API
        if (!('LanguageDetector' in self)) {
            errors.push("⚠️ A API de Detecção de Idioma não está ativa.");
            errors.push("Ative a seguinte flag em chrome://flags/:");
            errors.push("- Language Detection API (chrome://flags/#language-detector-api)");
        }

        if (errors.length > 0) {
            return errors;
        }

        const availability = await LanguageModel.availability({ languages: ["en"] });
        console.log('Language Model Availability:', availability);

        if (availability === 'available') {
            return null;
        }

        if (availability === 'unavailable') {
            errors.push(`⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA.`);
        }

        if (availability === 'downloading') {
            errors.push(`⚠️ O modelo de linguagem de IA está sendo baixado. Por favor, aguarde alguns minutos e tente novamente.`);
        }

        if (availability === 'downloadable') {
            errors.push(`⚠️ O modelo de linguagem de IA precisa ser baixado, baixando agora... (acompanhe o progresso no terminal do chrome)`);
            try {
                const session = await LanguageModel.create({
                    expectedInputLanguages: ["en"],
                    monitor(m) {
                        m.addEventListener('downloadprogress', (e) => {
                            const percent = ((e.loaded / e.total) * 100).toFixed(0);
                            console.log(`Downloaded ${percent}%`);
                        });
                    }
                });
                await session.prompt('Hello');
                session.destroy();

                // Re-check availability after download
                const newAvailability = await LanguageModel.availability({ languages: ["en"] });
                if (newAvailability === 'available') {
                    return null; // Download successful
                }
            } catch (error) {
                console.error('Error downloading model:', error);
                errors.push(`⚠️ Erro ao baixar o modelo: ${error.message}`);
            }
        }

        return errors.length > 0 ? errors : null;
    }

    async getParams() {
        try {
            // No seu Chrome Canary, LanguageModel não tem .params() nem .capabilities()
            // Então retornamos valores padrão para a interface (sliders) não quebrar
            return { 
                defaultTemperature: 1, 
                defaultTopK: 3, 
                maxTopK: 128, 
                maxTemperature: 2 
            };
        } catch (e) {
            return { defaultTemperature: 1, defaultTopK: 3 };
        }
    }

    async* createSession(question, temperature, topK, file = null) {
        this.abortController?.abort();
        this.abortController = new AbortController();

        if (this.session) {
            this.session.destroy();
        }

        // Adicionando expectedInputs para avisar ao Chrome que vamos enviar imagens
        this.session = await LanguageModel.create({
            temperature: temperature,
            topK: topK,
            expectedOutputLanguages: ["en"],
            expectedInputs: [
                { type: "text" },
                { type: "image" }
            ],
            initialPrompts: [
                {
                    role: 'system',
                    content: [{
                        type: "text",
                        value: `You are an AI assistant that responds clearly and objectively. Always respond in plain text.`
                    }]
                },
            ],
        });

        try {
            let prompt;

            if (file) {
                // Se houver arquivo, usamos o formato de array (multimodal)
                const fileType = file.type.split('/')[0];
                const blob = new Blob([await file.arrayBuffer()], { type: file.type });
                
                prompt = [
                    {
                        role: 'user',
                        content: [
                            { type: "text", value: question },
                            { type: fileType, value: blob }
                        ],
                    },
                ];
                console.log(`Enviando prompt multimodal com ${fileType}:`, file.name);
            } else {
                // Se for apenas texto, enviamos a string direta. 
                // Isso evita o erro "Required member is undefined" em algumas versões do Canary.
                prompt = question;
            }

            const responseStream = await this.session.promptStreaming(prompt, {
                signal: this.abortController.signal,
            });

            for await (const chunk of responseStream) {
                if (this.abortController.signal.aborted) break;
                yield chunk;
            }
        } catch (e) {
            console.error('Erro ao processar prompt:', e);
            yield `[Erro]: ${e.message}`;
        }
    }

    abort() {
        this.abortController?.abort();
    }

    isAborted() {
        return this.abortController?.signal.aborted;
    }
}
