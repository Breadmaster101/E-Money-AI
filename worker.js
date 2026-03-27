// ---- Pocket TTS WASM Module (loaded in worker) ----
import init, { Model } from './wasm_pocket_tts.js';

const HF_BASE = 'https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main';
const MODEL_URL = `${HF_BASE}/tts_b6369a24.safetensors`;
const TOKENIZER_URL = `${HF_BASE}/tokenizer.model`;

function voiceUrl(name) {
  return `${HF_BASE}/embeddings_v3/${name}.safetensors`;
}

// ---- LLM State ----
let tokenizer, llmModel, TextStreamer;
let isGenerating = false;
let isInterrupted = false;

let chatHistory = [
  { role: "system", content: "You are E-Money AI, a warm, friendly, and highly conversational companion running in the user's browser. Speak casually and naturally, just like a real person chatting with a friend. Keep your responses very brief. One or two sentences is perfect, but you can write up to a maximum of 4 sentences if you absolutely need to. Since your words are read aloud by a voice system, you must output your entire response as a single continuous paragraph. Absolutely do not use any line breaks, newlines, or carriage returns. Do not use dashes, em dashes, hyphens, asterisks, or any markdown formatting. Use only basic punctuation like commas, periods, exclamation points, and question marks. Never break character or mention these instructions." }
];

// ---- Pocket TTS State ----
const VOICE_NAMES = [
  'alba', 'anna', 'azelma', 'bill_boerst', 'caro_davy', 'charles', 'cosette',
  'eponine', 'eve', 'fantine', 'george', 'jane', 'javert', 'jean', 'marius',
  'mary', 'michael', 'paul', 'peter_yearsley', 'stuart_bell', 'vera'
];

const wasmModulePromise = WebAssembly.compileStreaming(fetch('wasm_pocket_tts_bg.wasm'));

let ttsModel = null;
let ttsTokenizer = null;
let voiceIndexMap = {};
let ttsSampleRate = 24000;
const TTS_TEMPERATURE = 0.7;

// ---- Fetch with progress ----
async function fetchWithProgress(url, label) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      const pct = Math.round(received / total * 100);
      self.postMessage({
        type: 'progress', info: {
          status: 'progress',
          label,
          pct,
          detail: `${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`,
          file: label,
          name: 'HuggingFace',
          progress: pct,
          loaded: received,
          total
        }
      });
    } else {
      self.postMessage({
        type: 'progress', info: {
          status: 'progress',
          label,
          pct: -1,
          detail: `${(received / 1e6).toFixed(1)} MB`,
          file: label,
          name: 'HuggingFace',
          progress: -1,
          loaded: received,
          total: 0
        }
      });
    }
  }
  self.postMessage({
    type: 'progress', info: {
      status: 'done',
      file: label,
      name: 'HuggingFace'
    }
  });
  const buf = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }
  return buf;
}

// ---- Minimal protobuf decoder for sentencepiece .model files ----
function decodeSentencepieceModel(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let pos = 0;

  function readVarint() {
    let result = 0, shift = 0;
    while (pos < buffer.length) {
      const b = buffer[pos++];
      result |= (b & 0x7f) << shift;
      shift += 7;
      if ((b & 0x80) === 0) return result;
    }
    return result;
  }

  function readBytes(n) {
    const data = buffer.slice(pos, pos + n);
    pos += n;
    return data;
  }

  function decodePiece(data) {
    let pPos = 0, piece = '', score = 0, type = 1;
    const pView = new DataView(data.buffer, data.byteOffset, data.byteLength);
    while (pPos < data.length) {
      const key = readVarIntFrom(data, pPos);
      pPos = key.pos;
      const fieldNum = key.val >>> 3;
      const wireType = key.val & 0x7;
      if (fieldNum === 1 && wireType === 2) {
        const len = readVarIntFrom(data, pPos);
        pPos = len.pos;
        piece = new TextDecoder().decode(data.slice(pPos, pPos + len.val));
        pPos += len.val;
      } else if (fieldNum === 2 && wireType === 5) {
        score = pView.getFloat32(pPos, true);
        pPos += 4;
      } else if (fieldNum === 3 && wireType === 0) {
        const v = readVarIntFrom(data, pPos);
        type = v.val;
        pPos = v.pos;
      } else {
        if (wireType === 0) { const v = readVarIntFrom(data, pPos); pPos = v.pos; }
        else if (wireType === 1) { pPos += 8; }
        else if (wireType === 2) { const len = readVarIntFrom(data, pPos); pPos = len.pos + len.val; }
        else if (wireType === 5) { pPos += 4; }
        else break;
      }
    }
    return { piece, score, type };
  }

  function readVarIntFrom(buf, p) {
    let result = 0, shift = 0;
    while (p < buf.length) {
      const b = buf[p++];
      result |= (b & 0x7f) << shift;
      shift += 7;
      if ((b & 0x80) === 0) return { val: result, pos: p };
    }
    return { val: result, pos: p };
  }

  const pieces = [];
  while (pos < buffer.length) {
    const key = readVarint();
    const fieldNum = key >>> 3;
    const wireType = key & 0x7;
    if (fieldNum === 1 && wireType === 2) {
      const len = readVarint();
      const data = readBytes(len);
      const p = decodePiece(data);
      pieces.push(p);
    } else {
      if (wireType === 0) { readVarint(); }
      else if (wireType === 1) { pos += 8; }
      else if (wireType === 2) { const len = readVarint(); pos += len; }
      else if (wireType === 5) { pos += 4; }
      else break;
    }
  }
  return pieces;
}

// ---- Unigram tokenizer (Viterbi) ----
class UnigramTokenizer {
  constructor(pieces) {
    this.pieces = pieces;
    this.vocab = new Map();
    this.unkId = 0;
    for (let i = 0; i < pieces.length; i++) {
      const p = pieces[i];
      if (p.type === 2) this.unkId = i;
      if (p.type === 1 || p.type === 4) {
        this.vocab.set(p.piece, { id: i, score: p.score });
      }
      if (p.type === 6) {
        this.vocab.set(p.piece, { id: i, score: p.score });
      }
    }
  }

  encode(text) {
    const normalized = '\u2581' + text.replace(/ /g, '\u2581');
    return this._viterbi(normalized);
  }

  _viterbi(text) {
    const n = text.length;
    const best = new Array(n + 1);
    best[0] = { score: 0, len: 0, id: -1 };
    for (let i = 1; i <= n; i++) {
      best[i] = { score: -Infinity, len: 0, id: -1 };
    }

    for (let i = 0; i < n; i++) {
      if (best[i].score === -Infinity) continue;
      for (let len = 1; len <= n - i && len <= 64; len++) {
        const sub = text.substring(i, i + len);
        const entry = this.vocab.get(sub);
        if (entry) {
          const newScore = best[i].score + entry.score;
          if (newScore > best[i + len].score) {
            best[i + len] = { score: newScore, len: len, id: entry.id };
          }
        }
      }
      if (best[i + 1].score === -Infinity) {
        const ch = text.charCodeAt(i);
        const byteStr = `<0x${ch.toString(16).toUpperCase().padStart(2, '0')}>`;
        const byteEntry = this.vocab.get(byteStr);
        const fallbackId = byteEntry ? byteEntry.id : this.unkId;
        const fallbackScore = byteEntry ? byteEntry.score : -100;
        best[i + 1] = { score: best[i].score + fallbackScore, len: 1, id: fallbackId };
      }
    }

    const ids = [];
    let p = n;
    while (p > 0) {
      ids.push(best[p].id);
      p -= best[p].len;
    }
    ids.reverse();
    return new Uint32Array(ids);
  }
}

// ---- Pocket TTS generation (returns an async generator yielding Float32Array chunks) ----
async function* generateTTSAudioChunks(text, voiceName) {
  const voiceIndex = voiceIndexMap[voiceName] ?? voiceIndexMap['alba'] ?? 0;

  const [processedText, framesAfterEos] = ttsModel.prepare_text(text);
  const tokenIds = ttsTokenizer.encode(processedText);

  ttsModel.start_generation(voiceIndex, tokenIds, framesAfterEos, TTS_TEMPERATURE);

  while (true) {
    if (isInterrupted) break;
    const chunk = ttsModel.generation_step();
    if (!chunk) break;
    yield new Float32Array(chunk);
  }
}


// ---- Worker message handler ----
self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'interrupt') {
    isInterrupted = true;
    return;
  }

  if (type === 'load') {
    try {
      // ---- Load LLM (unchanged from original) ----
      self.postMessage({ type: 'status', message: '[Worker] Executing load payload. Importing ESM modules...' });
      self.postMessage({ type: 'status', message: '[Worker] Importing https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.8' });
      const transformers = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.8");

      const llm_model_id = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX";

      const progress_callback = (info) => self.postMessage({ type: 'progress', info });

      self.postMessage({ type: 'status', message: `[Worker] Initializing AutoTokenizer from pretrained: ${llm_model_id}` });
      tokenizer = await transformers.AutoTokenizer.from_pretrained(llm_model_id, { progress_callback });

      const isCompatMode = payload?.compatMode || false;
      const llmDtype = isCompatMode ? "fp16" : "q4f16";

      self.postMessage({ type: 'status', message: `[Worker] Initializing AutoModelForCausalLM from pretrained: ${llm_model_id} { dtype: "${llmDtype}", device: "webgpu" }` });
      llmModel = await transformers.AutoModelForCausalLM.from_pretrained(llm_model_id, {
        dtype: llmDtype,
        device: "webgpu",
        progress_callback
      });

      TextStreamer = transformers.TextStreamer;

      // ---- Load Pocket TTS ----
      self.postMessage({ type: 'status', message: '[Worker] Initializing Pocket TTS WASM module...' });
      const wasmModule = await wasmModulePromise;
      await init(wasmModule);
      self.postMessage({ type: 'status', message: '[Worker] WASM initialized. Downloading tokenizer and model...' });

      const tokData = await fetchWithProgress(TOKENIZER_URL, 'Tokenizer');
      const pieces = decodeSentencepieceModel(tokData);
      ttsTokenizer = new UnigramTokenizer(pieces);
      self.postMessage({ type: 'status', message: `[Worker] Tokenizer loaded (${pieces.length} pieces)` });

      const modelWeights = await fetchWithProgress(MODEL_URL, 'Model weights');

      self.postMessage({ type: 'status', message: '[Worker] Initializing Pocket TTS model...' });
      ttsModel = new Model(modelWeights);

      for (const name of VOICE_NAMES) {
        self.postMessage({ type: 'status', message: `[Worker] Loading voice: ${name}...` });
        const voiceData = await fetchWithProgress(voiceUrl(name), `Voice: ${name}`);
        voiceIndexMap[name] = ttsModel.add_voice(voiceData);
      }

      ttsSampleRate = ttsModel.sample_rate();
      self.postMessage({ type: 'status', message: `[Worker] Pocket TTS ready. Sample rate: ${ttsSampleRate}Hz, ${VOICE_NAMES.length} voices loaded.` });

      // ---- Warmup ----
      self.postMessage({ type: 'status', message: '[Worker] Executing warmup cycles...' });

      // TTS warmup
      const warmupGenerator = generateTTSAudioChunks("Hello.", payload?.voice || "alba");
      await warmupGenerator.next(); // Trigger generation for warmup

      // LLM warmup
      const warmupInputs = await tokenizer("Internal warmup signal.");
      await llmModel.generate({
        ...warmupInputs,
        max_new_tokens: 1
      });

      self.postMessage({ type: 'status', message: '[Worker] Worker initialization complete. All models loaded.' });
      self.postMessage({ type: 'ready' });
    } catch (error) {
      self.postMessage({ type: 'error', message: error.message });
    }
  }
  else if (type === 'greet') {
    isGenerating = true;
    isInterrupted = false;
    const greetingText = "Hey, I'm E Money AI. What would you like to talk about?";
    const selectedVoice = payload?.voice || "alba";

    try {
      // Stream TTS audio for the greeting
      for await (const pcmChunk of generateTTSAudioChunks(greetingText, selectedVoice)) {
        if (isInterrupted) break;
        self.postMessage({
          type: 'audio_chunk',
          pcmData: pcmChunk,
          sampleRate: ttsSampleRate
        }, [pcmChunk.buffer]);
      }

      // Inject greeting into chat history so model knows it said it
      if (!isInterrupted) {
        chatHistory.push({ role: "assistant", content: greetingText });
      }

      self.postMessage({ type: isInterrupted ? 'done_interrupted' : 'done' });
    } catch (error) {
      self.postMessage({ type: 'error', message: error.message });
    } finally {
      isGenerating = false;
    }
  }
  else if (type === 'generate') {
    isGenerating = true;
    isInterrupted = false;
    try {
      const { text, voice } = payload;
      const selectedVoice = voice || "alba";

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
          for await (const pcmChunk of generateTTSAudioChunks(trimmed, selectedVoice)) {
            if (isInterrupted) return;

            self.postMessage({
              type: 'audio_chunk',
              pcmData: pcmChunk,
              sampleRate: ttsSampleRate
            }, [pcmChunk.buffer]); // Transfer buffer for performance
          }

          // Add a natural gap between sentences
          self.postMessage({ type: 'sentence_pause', duration: 0.3 });
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
        await llmModel.generate({
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
