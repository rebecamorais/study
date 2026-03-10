export class TranslationService {
    constructor() {
        this.translator = null;
        this.languageDetector = null;
    }

    async initialize() {
        try {
            // Tentando inicializar. Se precisar de download, o Chrome pode dar NotAllowedError no onload
            this.translator = await Translator.create({
                sourceLanguage: 'en',
                targetLanguage: 'pt'
            });
            console.log('Translator initialized');

            this.languageDetector = await LanguageDetector.create();
            console.log('Language Detector initialized');

            return true;
        } catch (error) {
            if (error.name === 'NotAllowedError') {
                console.warn('Tradução precisa de interação do usuário para baixar. Use o botão "Forçar Download".');
                return false;
            }
            console.error('Error initializing translation:', error);
            // Em vez de dar throw e travar tudo, apenas retornamos false
            return false;
        }
    }

    async translateToPortuguese(text) {
        if (!this.translator) {
            console.warn('Translator not available, returning original text');
            return text;
        }

        try {
            // Detect language first
            if (this.languageDetector) {
                const detectionResults = await this.languageDetector.detect(text);
                console.log('Detected languages:', detectionResults);

                // If already in Portuguese, no need to translate
                if (detectionResults && detectionResults[0]?.detectedLanguage === 'pt') {
                    console.log('Text is already in Portuguese');
                    return text;
                }
            }

            // Use streaming translation
            const stream = this.translator.translateStreaming(text);
            let translated = '';
            for await (const chunk of stream) {
                translated = chunk; // Each chunk is the full translation so far
            }
            console.log('Translated text:', translated);
            return translated;
        } catch (error) {
            console.error('Translation error:', error);
            return text; // Return original text if translation fails
        }
    }
}
