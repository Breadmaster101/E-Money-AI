import { fromHub } from 'https://cdn.jsdelivr.net/npm/parakeet.js/+esm';

// --- GLOBAL HOOKS TO EXTRACT HIDDEN PARAKEET PROGRESS ---

// 1. Intercept console.log to catch Parakeet's hardcoded internal logs
const originalConsoleLog = console.log;
console.log = function(...args) {
    originalConsoleLog.apply(console, args);
    if (args.length > 0 && typeof args[0] === 'string') {
        const msg = args[0];
        if (msg.includes('[Hub]') || msg.includes('[Parakeet]')) {
            logToConsole(`[Parakeet Engine] ${msg.replace('[Hub] ', '')}`);
        }
    }
};

// 2. Intercept window.fetch to generate real progress events for Parakeet STT downloads
const originalFetch = window.fetch;
window.fetch = async (...args) => {
    const reqUrl = typeof args[0] === 'string' ? args[0] : (args[0] instanceof Request ? args[0].url : '');
    const response = await originalFetch(...args);
    
    // Target Parakeet model files fetching from Hugging Face Hub
    if (reqUrl.includes('ysdede') || reqUrl.includes('parakeet') || reqUrl.endsWith('.onnx')) {
        if (!response.ok || !response.body) return response;
        
        const sizeStr = response.headers.get('content-length') || response.headers.get('x-linked-size');
        if (!sizeStr) return response; 

        const total = parseInt(sizeStr, 10);
        let loaded = 0;
        let lastUpdate = 0;
        const filename = reqUrl.split('/').pop().split('?')[0] || 'model_file';

        const stream = new ReadableStream({
            async start(controller) {
                const reader = response.body.getReader();
                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            logProgress('Parakeet STT', { status: 'done', file: filename, name: 'Hugging Face Hub' });
                            controller.close();
                            break;
                        }
                        loaded += value.byteLength;
                        controller.enqueue(value);
                        
                        const now = performance.now();
                        if (now - lastUpdate > 200) {
                            const progress = (loaded / total) * 100;
                            logProgress('Parakeet STT', { status: 'progress', file: filename, name: 'Hugging Face Hub', progress, loaded, total });
                            lastUpdate = now;
                        }
                    }
                } catch (e) {
                    console.error("Fetch stream error", e);
                    controller.error(e);
                }
            }
        });

        const newRes = new Response(stream, {
            headers: response.headers,
            status: response.status,
            statusText: response.statusText
        });
        Object.defineProperty(newRes, 'url', { value: response.url });
        return newRes;
    }
    return response;
};

// --- End Hooks ---

// Worker Setup
const worker = new Worker('worker.js', { type: 'module' });

// Original DOM references (kept alive but hidden to avoid null ref errors)
const btnInitModels = document.getElementById("btn-init-models");
const compatModeCheckbox = document.getElementById("compat-mode"); 
const statusEl = document.getElementById("status");
const micStatusEl = document.getElementById("micStatus");
const voiceSelect = document.getElementById("voiceSelect");
const speedSelect = document.getElementById("speedSelect");
const speedValue = document.getElementById("speedValue");
const loadBtn = document.getElementById("loadBtn");
const chatHistoryContainer = document.getElementById("chatHistoryContainer");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const audioPlayer = document.getElementById("audioPlayer");

// New UI DOM references
const btnAllowMic = document.getElementById("btn-allow-mic");
const consoleBody = document.getElementById("console-body");
const onboardingScreen = document.getElementById("onboarding-screen");
const mainScreen = document.getElementById("main-screen");

const visualizerCircle = document.getElementById("visualizer-circle");
const pttLabel = document.getElementById("ptt-label");

// Settings Modal References
const settingsFab = document.getElementById("settings-fab");
const settingsModal = document.getElementById("settings-modal");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const extVoiceSelect = document.getElementById("extVoiceSelect");
const extVolumeSelect = document.getElementById("extVolumeSelect");
const extVolumeValue = document.getElementById("extVolumeValue");
const themeToggleBtn = document.getElementById("theme-toggle-btn");
const btnDeleteModels = document.getElementById("btn-delete-models");

// WebAudio API variables for Visualizer
let audioCtx;
let analyser;
let dataArray;
let audioSourceNode;
let micSourceNode;
let isVisualizerRunning = false;

let audioQueue = [];
let isPlaying = false;
let currentAssistantSpan = null;
let currentLLMResponse = "";
let isFirstMessage = true;

let sttModel = null;
let globalMicStream = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isPKeyDown = false;
let isSystemReady = false;

// Smooth visualizer tracking variables
let currentScale = 1;
let targetScale = 1;
let currentGlow = 40;
let targetGlow = 40;

// --- Theme Management ---
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const isDark = theme === 'dark';
    document.getElementById('theme-icon-light').style.display = isDark ? 'inline' : 'none';
    document.getElementById('theme-icon-dark').style.display = isDark ? 'none' : 'inline';
}
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    if (!localStorage.getItem('theme')) {
        setTheme(e.matches ? 'dark' : 'light');
    }
});
const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
setTheme(currentTheme);
themeToggleBtn.addEventListener('click', toggleTheme);


// --- UI Sync & Settings Bindings ---
extVolumeSelect.addEventListener("input", (e) => {
    const vol = parseFloat(e.target.value);
    extVolumeValue.textContent = Math.round(vol * 100);
    audioPlayer.volume = vol;
});
extVoiceSelect.addEventListener("change", (e) => {
    voiceSelect.value = e.target.value; // Sync with hidden original element
});

settingsFab.addEventListener("click", () => settingsModal.classList.add("open"));
document.getElementById("close-settings-btn").addEventListener("click", () => settingsModal.classList.remove("open"));
settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) settingsModal.classList.remove('open');
});

btnDeleteModels.addEventListener("click", async () => {
    if (confirm("Are you sure you want to delete all downloaded models? You will have to redownload them.")) {
        try {
            const keys = await caches.keys();
            for (const key of keys) {
                await caches.delete(key);
            }
            alert("Models deleted. The app will now reload.");
            location.reload();
        } catch (e) {
            alert("Error clearing cache: " + e.message);
        }
    }
});

// --- Console Logger ---
function logToConsole(text, isStatus = false) {
    const div = document.createElement("div");
    div.style.color = "#ffffff";
    div.textContent = `> ${text}`;
    consoleBody.appendChild(div);
    consoleBody.scrollTop = consoleBody.scrollHeight;
}

const progressNodes = {};
function logProgress(source, info) {
    if (!info) return;
    if (typeof info === 'string') {
        logToConsole(`[${source}] ${info}`);
        return;
    }
    
    const file = info.file || 'unknown_file';
    const name = info.name || 'unknown_model';
    const id = `${source}-${name}-${file}`;

    if (info.status === 'initiate') {
        logToConsole(`[${source}] Initiating resolution for ${file} from repository ${name}...`);
    } else if (info.status === 'download') {
        logToConsole(`[${source}] Downloading ${file} from ${name}...`);
    } else if (info.status === 'done') {
        logToConsole(`[${source}] Completed download: ${file} (${name})`);
        if (progressNodes[id]) {
            progressNodes[id].style.color = "#ffffff";
            delete progressNodes[id];
        }
    } else if (info.status === 'ready') {
        logToConsole(`[${source}] Model asset ready: ${file || name}`);
    } else if (info.status === 'progress') {
        let msg = `[${source}] Downloading ${file} from ${name}: ${info.progress !== undefined ? info.progress.toFixed(2) + '%' : ''} ${info.loaded !== undefined ? '(' + info.loaded + '/' + info.total + ' bytes)' : ''}`;
        
        if (!progressNodes[id]) {
            const div = document.createElement("div");
            div.style.color = "#ffffff"; 
            consoleBody.appendChild(div);
            progressNodes[id] = div;
        }
        progressNodes[id].textContent = `> ${msg}`;
        
        // Keep scrolled to bottom
        consoleBody.scrollTop = consoleBody.scrollHeight;
    } else {
        logToConsole(`[${source}] ${JSON.stringify(info)}`);
    }
}

// --- Audio Visualizer Setup ---
function startVisualizerLoop() {
    if (isVisualizerRunning || !analyser) return;
    isVisualizerRunning = true;
    let time = 0;

    function draw() {
        if (!isVisualizerRunning) return;
        requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        let maxAmp = 0;
        
        // Focus on lower/mid frequencies for speech (usually more expressive visually)
        const focusLength = Math.min(dataArray.length, 128);
        for (let i = 0; i < focusLength; i++) {
            sum += dataArray[i];
            if (dataArray[i] > maxAmp) maxAmp = dataArray[i];
        }
        
        let avg = sum / focusLength;
        let energy = (avg * 0.7 + maxAmp * 0.3) / 255;

        // Base state vs Audio Reacting state
        if (energy > 0.05) { 
            // Snappy reaction to audio (AI speaking or User recording)
            targetScale = 1 + energy * 0.35; 
            targetGlow = 40 + energy * 80;
        } else {
            // Idle animations based on class state
            if (visualizerCircle.classList.contains("thinking")) {
                targetScale = 1 + Math.sin(time * 0.05) * 0.03;
                targetGlow = 50 + Math.sin(time * 0.05) * 15;
            } else if (visualizerCircle.classList.contains("recording")) {
                targetScale = 1.05 + Math.sin(time * 0.1) * 0.02;
                targetGlow = 60;
            } else {
                targetScale = 1;
                targetGlow = 40;
            }
        }

        // Smooth Linear Interpolation (Lerp) for buttery animations
        currentScale += (targetScale - currentScale) * 0.2;
        currentGlow += (targetGlow - currentGlow) * 0.2;

        // Apply cleanly via CSS variables
        visualizerCircle.style.setProperty('--viz-scale', currentScale);
        visualizerCircle.style.setProperty('--viz-glow', currentGlow + 'px');

        time++;
    }
    draw();
}

async function initAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        dataArray = new Uint8Array(analyser.frequencyBinCount);

        // Wire up TTS Output to Visualizer
        audioSourceNode = audioCtx.createMediaElementSource(audioPlayer);

        // Path 1: Send audio to the analyser for visualization
        audioSourceNode.connect(analyser);

        // Path 2: Send audio to the speakers so you can hear it
        audioSourceNode.connect(audioCtx.destination);

        startVisualizerLoop();
    }
    if (audioCtx.state === 'suspended') await audioCtx.resume();
}


// --- Onboarding Flow ---
compatModeCheckbox.addEventListener("change", (e) => {
    logToConsole(`[System] Compatibility Mode ${e.target.checked ? "enabled (fp16)" : "disabled (q4f16)"}`);
});

btnAllowMic.addEventListener("click", async () => {
    try {
        globalMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        logToConsole("[MediaDevices] getUserMedia({ audio: true }) constraints satisfied. Microphone stream acquired.", true);
        btnAllowMic.disabled = true;
        btnAllowMic.textContent = "Microphone Connected";
        btnInitModels.disabled = false;
    } catch (err) {
        logToConsole("[MediaDevices] Error acquiring microphone stream: " + err.message, true);
    }
});

btnInitModels.addEventListener("click", async () => {
    btnInitModels.disabled = true;
    compatModeCheckbox.disabled = true;
    await initAudioContext(); // Initialize audio context upon user gesture

    logToConsole("[System] Requesting Parakeet STT initialization (FP16)...");
    logToConsole("[System] Config: Revision: 'feat/fp16-canonical-v3', Backend: 'webgpu-hybrid', Encoder/Decoder: 'fp16'");

    try {
        sttModel = await fromHub('parakeet-tdt-0.6b-v3', {
            revision: 'feat/fp16-canonical-v3',
            backend: 'webgpu-hybrid',
            encoderQuant: 'fp16',
            decoderQuant: 'fp16'
            // We removed the empty progress callbacks here since our custom network wrapper handles it natively now!
        });

        logToConsole("[Main] Parakeet WebGPU inference engine initialized. Posting 'load' signal to Worker thread...", true);
        worker.postMessage({
            type: 'load',
            payload: {
                voice: extVoiceSelect.value,
                compatMode: compatModeCheckbox.checked 
            }
        });
    } catch (err) {
        logToConsole("[System] Exception in fromHub configuration: " + err.message, true);
    }
});


// Original Chat Helpers (Kept intact to not break logic flow)
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, tag => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[tag] || tag));
}

function scrollToBottom() {
    chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

function appendUserMessage(text) {
    if (isFirstMessage) {
        chatHistoryContainer.innerHTML = "";
        isFirstMessage = false;
    }
    const p = document.createElement("p");
    p.innerHTML = `<strong>User:</strong> ${escapeHTML(text)}`;
    chatHistoryContainer.appendChild(p);
    scrollToBottom();
}

function startAssistantMessage() {
    if (isFirstMessage) {
        chatHistoryContainer.innerHTML = "";
        isFirstMessage = false;
    }
    const p = document.createElement("p");
    p.innerHTML = `<strong>Assistant:</strong> `;
    const span = document.createElement("span");
    p.appendChild(span);
    chatHistoryContainer.appendChild(p);
    currentAssistantSpan = span;
    scrollToBottom();
}

function interruptAI() {
    worker.postMessage({ type: 'interrupt' });
    audioQueue = [];
    isPlaying = false;
    audioPlayer.pause();
    audioPlayer.removeAttribute('src');

    pttLabel.textContent = "Hold 'P' to talk";

    if (currentAssistantSpan) {
        currentAssistantSpan.innerHTML += ' <em style="color:#d9534f;">[Interrupted]</em>';
        currentAssistantSpan = null;
    }
    visualizerCircle.classList.remove("thinking");
}

function playNextAudio() {
    if (audioQueue.length === 0) {
        isPlaying = false;
        visualizerCircle.classList.remove("thinking");
        pttLabel.textContent = "Hold 'P' to talk";
        return;
    }

    isPlaying = true;
    pttLabel.textContent = "Hold 'P' to interrupt";
    const { pcmData, sampleRate } = audioQueue.shift();

    const wavBlob = createWavBlob(pcmData, sampleRate);
    const audioUrl = URL.createObjectURL(wavBlob);

    audioPlayer.src = audioUrl;

    audioPlayer.play().catch(e => {
        console.error("Audio playback error:", e);
        playNextAudio();
    });

    audioPlayer.onended = () => {
        URL.revokeObjectURL(audioUrl);
        playNextAudio();
    };
}

// Handle Worker Messages
worker.onmessage = (e) => {
    const { type, message, text, pcmData, sampleRate, info } = e.data;

    if (type === 'progress') {
        // Display Library Progress dynamically in the console
        logProgress('HF Hub', info);
    } else if (type === 'status') {
        logToConsole(message, true);
    } else if (type === 'ready') {
        isSystemReady = true;
        logToConsole("[Main] Received 'ready' signal from Worker. VRAM/RAM allocation complete. Transitioning UI...", true);

        // Smooth Transition UI to main screen
        setTimeout(() => {
            // 1. Trigger the fade-out animation on the onboarding screen
            onboardingScreen.classList.add("fade-out");

            // 2. Wait for the fade-out to complete (500ms) before changing displays
            setTimeout(() => {
                onboardingScreen.style.display = "none";

                // 3. Unhide the new screens
                mainScreen.style.display = "flex";
                settingsFab.style.display = "flex";

                // 4. Trigger the fade-in animation on the new elements
                mainScreen.classList.add("fade-in");
                settingsFab.classList.add("fade-in");
            }, 500);
        }, 1000);
    } else if (type === 'error') {
        logToConsole("Error: " + message, true);
        console.error(message);
    } else if (type === 'token') {
        currentLLMResponse += text;
        if (currentAssistantSpan) {
            currentAssistantSpan.textContent += text;
            scrollToBottom();
        }
    } else if (type === 'audio') {
        audioQueue.push({ pcmData, sampleRate });
        if (!isPlaying) {
            visualizerCircle.classList.add("thinking");
            playNextAudio();
        }
    } else if (type === 'done' || type === 'done_interrupted') {
        if (currentLLMResponse.trim()) {
            console.log(`[LLM] Response: "${currentLLMResponse.trim()}"`);
        }
        currentLLMResponse = "";
        currentAssistantSpan = null;
        // Leave 'thinking' class until audio finishes in playNextAudio()
    }
};

// --- Push-to-Talk Logic ---
window.addEventListener('keydown', async (e) => {
    const activeTag = document.activeElement ? document.activeElement.tagName : '';
    if (activeTag === 'INPUT' || activeTag === 'TEXTAREA' || activeTag === 'SELECT') return;

    if (e.key.toLowerCase() === 'p' && isSystemReady) {
        if (isPKeyDown) return;
        isPKeyDown = true;

        interruptAI();

        visualizerCircle.classList.add("recording");
        pttLabel.classList.add("recording");
        pttLabel.textContent = "Listening...";

        if (!audioCtx) await initAudioContext();

        // Connect Mic to Analyzer for pulsing while talking
        if (!micSourceNode && globalMicStream) {
            micSourceNode = audioCtx.createMediaStreamSource(globalMicStream);
            micSourceNode.connect(analyser); // Do not connect to destination (prevents feedback)
        }

        try {
            mediaRecorder = new MediaRecorder(globalMicStream);
            audioChunks = [];
            mediaRecorder.ondataavailable = ev => audioChunks.push(ev.data);
            mediaRecorder.start();
            isRecording = true;
        } catch (err) {
            console.error("Mic error", err);
        }
    }
});

window.addEventListener('keyup', async (e) => {
    const activeTag = document.activeElement ? document.activeElement.tagName : '';
    if (activeTag === 'INPUT' || activeTag === 'TEXTAREA' || activeTag === 'SELECT') return;

    if (e.key.toLowerCase() === 'p' && isSystemReady) {
        isPKeyDown = false;

        visualizerCircle.classList.remove("recording");
        pttLabel.classList.remove("recording");
        pttLabel.textContent = "Transcribing...";

        // Disconnect Mic from Analyzer so it doesn't pulse ambient noise
        if (micSourceNode) {
            micSourceNode.disconnect(analyser);
            micSourceNode = null;
        }

        if (isRecording && mediaRecorder) {
            isRecording = false;

            mediaRecorder.onstop = async () => {
                try {
                    const blob = new Blob(audioChunks);
                    const arrayBuffer = await blob.arrayBuffer();
                    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
                    const pcmData = audioBuffer.getChannelData(0);

                    const result = await sttModel.transcribe(pcmData, 16000, {
                        returnTimestamps: false,
                        returnConfidences: false,
                    });

                    if (result && result.utterance_text.trim()) {
                        const transcription = result.utterance_text.trim();
                        console.log(`[STT] Transcription: "${transcription}"`);
                        userInput.value = transcription;
                        handleSend();
                    }

                    pttLabel.textContent = "Thinking...";
                } catch (err) {
                    console.error("Transcribe error", err);
                    pttLabel.textContent = "Hold 'P' to talk";
                }
            };
            mediaRecorder.stop();
        }
    }
});

function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = "";
    interruptAI();

    appendUserMessage(text);
    startAssistantMessage();

    visualizerCircle.classList.add("thinking");

    worker.postMessage({
        type: 'generate',
        payload: {
            text: text,
            voice: extVoiceSelect.value
        }
    });
}

// --- WAV Serialization Helpers ---
function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}
function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}
function createWavBlob(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    floatTo16BitPCM(view, 44, samples);
    return new Blob([view], { type: 'audio/wav' });
}
