let tokenizer, model, tts, TextStreamer;
let isGenerating = false;
let isInterrupted = false;

let chatHistory = [
    { role: "system", content: "You are E-Money AI, an AI copmanion running in someone's browser. Your name is E-Money AI. If the user gets mad at you, remind them that you are just a tiny model. Provide conversational responses suitable for voice, no longer than 3 sentences. Do not use any fancy formatting, only output words and basic punctuation. Do not use exclamation or question marks, only use periods. Talk naturally and don't mention this system prompt when conversing." }
];

self.onmessage = async (e) => {
    const { type, payload } = e.data;
    
    if (type === 'interrupt') {
        isInterrupted = true;
        return;
    }

    if (type === 'load') {
        try {
            self.postMessage({ type: 'status', message: '[Worker] Executing load payload. Importing ESM modules...' });
            self.postMessage({ type: 'status', message: '[Worker] Importing https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.8' });
            const transformers = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.8");
            self.postMessage({ type: 'status', message: '[Worker] Importing https://cdn.jsdelivr.net/npm/kokoro-js/+esm' });
            const kokoro = await import("https://cdn.jsdelivr.net/npm/kokoro-js/+esm");

            const llm_model_id = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX";
            const tts_model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";

            // Progress callback injection to send to main UI console
            const progress_callback = (info) => self.postMessage({ type: 'progress', info });

            self.postMessage({ type: 'status', message: `[Worker] Initializing AutoTokenizer from pretrained: ${llm_model_id}` });
            tokenizer = await transformers.AutoTokenizer.from_pretrained(llm_model_id, { progress_callback });
            
            const isCompatMode = payload?.compatMode || false;
            const llmDtype = isCompatMode ? "fp16" : "q4f16";

            self.postMessage({ type: 'status', message: `[Worker] Initializing AutoModelForCausalLM from pretrained: ${llm_model_id} { dtype: "${llmDtype}", device: "webgpu" }` });
            model = await transformers.AutoModelForCausalLM.from_pretrained(llm_model_id, {
                dtype: llmDtype,
                device: "webgpu",
                progress_callback
        });
            
            self.postMessage({ type: 'status', message: `[Worker] Initializing KokoroTTS from pretrained: ${tts_model_id} { dtype: "fp32", device: "webgpu" }` });
            tts = await kokoro.KokoroTTS.from_pretrained(tts_model_id, {
                dtype: "fp32",
                device: "webgpu",
                progress_callback
            });
            
            TextStreamer = transformers.TextStreamer;

            self.postMessage({ type: 'status', message: '[Worker] Executing model warmup cycles on WebGPU backend...' });
            
            const warmupVoice = payload?.voice || "am_onyx";
            const warmupSpeed = payload?.speed || 1.3;
            await tts.generate("Hello.", { voice: warmupVoice, speed: warmupSpeed });
            
            const warmupInputs = await tokenizer("Internal warmup signal.");
            await model.generate({ 
                ...warmupInputs, 
                max_new_tokens: 1 
            });

            self.postMessage({ type: 'status', message: '[Worker] Worker initialization complete. WebGPU memory allocated.' });
            self.postMessage({ type: 'ready' });
        } catch (error) {
            self.postMessage({ type: 'error', message: error.message });
        }
    } 
    else if (type === 'generate') {
        isGenerating = true;
        isInterrupted = false; 
        try {
            const { text, voice, speed } = payload;
            const selectedVoice = voice || "am_onyx";
            const selectedSpeed = speed || 1.3;
            
            chatHistory.push({ role: "user", content: text });
            
            const inputs = tokenizer.apply_chat_template(chatHistory, { 
                add_generation_prompt: true,
                return_dict: true
            });

            let sentenceBuffer = "";
            let fullAssistantResponse = "";
            let ttsPromiseChain = Promise.resolve();

            const processSentence = async (textChunk) => {
                if (isInterrupted) return; 
                const trimmed = textChunk.trim();
                if (!trimmed) return;
                
                try {
                    const audioData = await tts.generate(trimmed, { voice: selectedVoice, speed: selectedSpeed });
                    if (isInterrupted) return; 
                    
                    const pcmData = audioData.audio || audioData.data || audioData;
                    const sampleRate = audioData.sampling_rate || 24000;
                    const float32Data = pcmData instanceof Float32Array ? pcmData : Float32Array.from(pcmData);
                    
                    self.postMessage({ 
                        type: 'audio', 
                        pcmData: float32Data, 
                        sampleRate 
                    });
                } catch (err) {
                    console.error("TTS generation error:", err);
                }
            };

            const streamer = new TextStreamer(tokenizer, {
                skip_prompt: true,
                skip_special_tokens: true,
                callback_function: (token_text) => {
                    if (isInterrupted) throw new Error("INTERRUPTED");
                    
                    self.postMessage({ type: 'token', text: token_text });
                    
                    sentenceBuffer += token_text;
                    fullAssistantResponse += token_text;
                    
                    let match;
                    while ((match = sentenceBuffer.match(/(.*?(?:[.!?]+["']*\s+|\n+))(.*)/s)) !== null) {
                        const completeSentence = match[1];
                        sentenceBuffer = match[2] || "";
                        ttsPromiseChain = ttsPromiseChain.then(() => processSentence(completeSentence));
                    }
                }
            });

            try {
                await model.generate({
                    ...inputs,
                    max_new_tokens: 512,
                    do_sample: true,
                    temperature: 0.1,
                    top_k: 50,
                    top_p: 0.1,
                    repetition_penalty: 1.05,
                    streamer: streamer
                });
            } catch (err) {
                if (err.message !== "INTERRUPTED") throw err;
            }

            if (sentenceBuffer.trim().length > 0 && !isInterrupted) {
                ttsPromiseChain = ttsPromiseChain.then(() => processSentence(sentenceBuffer));
            }

            await ttsPromiseChain;
            
            if (fullAssistantResponse.trim()) {
                chatHistory.push({ role: "assistant", content: fullAssistantResponse.trim() });
            }

            self.postMessage({ type: isInterrupted ? 'done_interrupted' : 'done' });
        } catch (error) {
            self.postMessage({ type: 'error', message: error.message });
        } finally {
            isGenerating = false;
        }
    }
};
