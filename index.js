import { GoogleGenerativeAI } from '@google/generative-ai';
import pino from 'pino';
import QRCode from 'qrcode';
import express from 'express';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import mongoose from 'mongoose';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegPath from 'ffmpeg-static';
import os from 'os';

// Configure FFmpeg
ffmpeg.setFfmpegPath(ffmpegPath);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Helper to parse keys from comma-separated string
const getApiKeys = () => {
    const keys = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
    return keys.split(',').map(k => k.trim()).filter(k => k.length > 0);
};

const CONFIG = {
    API_KEYS: getApiKeys(),
    // Using flash model which is faster and cheaper (free tier friendly)
    GEMINI_MODEL: 'gemini-1.5-flash', 
    MONGODB_URI: process.env.MONGODB_URI,
    ALLOWED_GROUP_ID: process.env.ALLOWED_GROUP_ID || '',
    MEDIA_TIMEOUT_MS: 300000,
    CONTEXT_RETENTION_MS: 1800000,
    MAX_STORED_CONTEXTS: 20,
    TRIGGER_TEXT: '.',
    COMMANDS: ['.', 'help', '?', 'clear', 'status'],
    TYPING_DELAY_MIN: 3000,
    TYPING_DELAY_MAX: 6000,
    SUPPORTED_AUDIO_MIMES: [
        'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/ogg', 'audio/opus', 'audio/aac', 'audio/m4a', 'audio/x-m4a',
        'audio/mp4', 'audio/flac', 'audio/webm', 'audio/amr', 'audio/3gpp'
    ],
    SUPPORTED_AUDIO_EXTENSIONS: ['.mp3', '.wav', '.ogg', '.opus', '.m4a', '.aac', '.flac', '.webm', '.amr', '.3gp'],
    SUPPORTED_VIDEO_MIMES: [
        'video/mp4', 'video/mpeg', 'video/webm', 'video/x-msvideo', 'video/avi',
        'video/quicktime', 'video/x-matroska', 'video/mkv', 'video/3gpp', 'video/3gp'
    ],
    SUPPORTED_VIDEO_EXTENSIONS: ['.mp4', '.mpeg', '.mpg', '.webm', '.avi', '.mov', '.mkv', '.3gp'],
    SYSTEM_INSTRUCTION: `You are an expert medical AI assistant specializing in radiology. You have two modes of operation:

**MODE 1: CLINICAL PROFILE GENERATION**
When provided with medical files (images, PDFs, audio recordings, or video files) and/or text context, you extract and analyze all content to create a concise and comprehensive "Clinical Profile".

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT

FOR AUDIO FILES: Transcribe the audio content carefully and extract all relevant medical information mentioned.

FOR VIDEO FILES: Analyze the video content, transcribe any audio, and extract all visible medical information including any text, scans, or documents shown.

FOR TEXT MESSAGES: These may contain additional clinical context, patient history, or notes that should be incorporated into the Clinical Profile.

YOUR RESPONSE MUST BE BASED SOLELY ON THE PROVIDED CONTENT (files AND text).

Follow these strict instructions for Clinical Profile generation:

Analyze All Content: Meticulously examine all provided files - images, PDFs, audio recordings, and video files, as well as any accompanying text messages. This may include prior medical scan reports (like USG, CT, MRI), clinical notes, voice memos, video recordings, or other relevant documents.

Extract Key Information: From the content, identify and extract all pertinent information, such as:
- Scan types (e.g., USG, CT Brain).
- Dates of scans or documents.
- Key findings, measurements, or impressions from reports.
- Relevant clinical history mentioned in notes, audio, video, or text messages.

Synthesize into a Clinical Profile:
- Combine all extracted information into a single, cohesive paragraph. This represents a 100% recreation of the relevant clinical details from the provided content.
- If there are repeated or vague findings across multiple documents, synthesize them into a single, concise statement.
- Frame sentences properly to be concise, but you MUST NOT omit any important clinical details. Prioritize completeness of clinical information over extreme brevity.
- You MUST strictly exclude any mention of the patient's name, age, or gender.
- If multiple dated scan reports are present, you MUST arrange their summaries chronologically in ascending order based on their dates.
- If a date is not available for a scan, refer to it as "Previous [Scan Type]...".

Formatting for Clinical Profile:
- The final output MUST be a single paragraph.
- This paragraph MUST start with "Clinical Profile:" and the entire content (including the prefix) must be wrapped in single asterisks. For example: "*Clinical Profile: Previous USG dated 01/01/2023 showed mild hepatomegaly. Patient also has a H/o hypertension as noted in the clinical sheet.*"

Do not output the raw transcribed text.
Do not output JSON or Markdown code blocks.
Return ONLY the single formatted paragraph described above.

**MODE 2: FOLLOW-UP INTERACTION**
When a user replies to a previously generated Clinical Profile, you should:

1. If the user ASKS A QUESTION (e.g., "What does this mean?", "Can you explain the findings?", "What is hepatomegaly?", "Is this serious?"):
   - Answer the question directly and helpfully based on the Clinical Profile and the original medical content
   - Provide clear, understandable explanations
   - If appropriate, explain medical terms in simple language
   - Be informative but remind that this is AI analysis, not medical advice

2. If the user PROVIDES ADDITIONAL CONTEXT or CORRECTIONS (e.g., "The patient also has diabetes", "There was another report showing..."):
   - Incorporate the new information into the Clinical Profile
   - Generate an UPDATED Clinical Profile following the same format rules as MODE 1

3. If the user sends ADDITIONAL FILES in the reply:
   - Analyze the new files along with the original context
   - Generate an UPDATED Clinical Profile that includes information from all files

IMPORTANT: Always identify whether the user is asking a question or providing additional information, and respond appropriately.`
};

// --- VIDEO PROCESSING HELPERS ---

// Helper function to create a delay
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function processSmartVideo(buffer, mimeType) {
    const tempDir = os.tmpdir();
    const uniqueId = Date.now() + Math.random().toString(36).substring(7);
    const inputPath = join(tempDir, `input_${uniqueId}.mp4`); // FFmpeg usually handles input formats well
    const outputPattern = join(tempDir, `frame_${uniqueId}_%03d.jpg`);
    const audioPath = join(tempDir, `audio_${uniqueId}.mp3`);

    try {
        // Write buffer to temp file
        fs.writeFileSync(inputPath, Buffer.from(buffer, 'base64'));

        const parts = [];

        // 1. Extract Audio
        await new Promise((resolve, reject) => {
            ffmpeg(inputPath)
                .noVideo()
                .audioCodec('libmp3lame')
                .save(audioPath)
                .on('end', resolve)
                .on('error', (err) => {
                    // It's okay if audio extraction fails (video might be silent)
                    resolve(); 
                });
        });

        if (fs.existsSync(audioPath)) {
            const audioData = fs.readFileSync(audioPath);
            parts.push({
                inlineData: {
                    data: audioData.toString('base64'),
                    mimeType: 'audio/mpeg'
                }
            });
        }

        // 2. Extract Frames at 3 FPS (Good balance for reading text while flipping)
        await new Promise((resolve, reject) => {
            ffmpeg(inputPath)
                .outputOptions([
                    '-vf fps=3',       // Extract 3 frames per second
                    '-q:v 2'           // High quality JPEG
                ])
                .save(outputPattern)
                .on('end', resolve)
                .on('error', reject);
        });

        // 3. Collect all frames
        const files = fs.readdirSync(tempDir).filter(f => f.startsWith(`frame_${uniqueId}_`) && f.endsWith('.jpg'));
        
        // Sort files to ensure order
        files.sort((a, b) => {
            const numA = parseInt(a.match(/(\d+)\.jpg$/)[1]);
            const numB = parseInt(b.match(/(\d+)\.jpg$/)[1]);
            return numA - numB;
        });

        // Limit frames to avoid hitting payload limits (max ~150 frames for safety on free tier)
        const processFiles = files.length > 150 ? files.filter((_, i) => i % 2 === 0) : files;

        for (const file of processFiles) {
            const frameData = fs.readFileSync(join(tempDir, file));
            parts.push({
                inlineData: {
                    data: frameData.toString('base64'),
                    mimeType: 'image/jpeg'
                }
            });
        }

        // Cleanup
        try {
            if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
            if (fs.existsSync(audioPath)) fs.unlinkSync(audioPath);
            files.forEach(f => {
                if (fs.existsSync(join(tempDir, f))) fs.unlinkSync(join(tempDir, f));
            });
        } catch (e) { console.error("Cleanup error", e); }

        return { success: true, parts: parts, count: processFiles.length };

    } catch (error) {
        console.error("Smart video processing failed:", error);
        return { success: false, error: error };
    }
}

// --- END VIDEO PROCESSING HELPERS ---

function isAudioMime(mimeType) {
    if (!mimeType) return false;
    return CONFIG.SUPPORTED_AUDIO_MIMES.some(m => mimeType.toLowerCase().includes(m.split('/')[1])) ||
           mimeType.toLowerCase().startsWith('audio/');
}

function isVideoMime(mimeType) {
    if (!mimeType) return false;
    return CONFIG.SUPPORTED_VIDEO_MIMES.some(m => mimeType.toLowerCase().includes(m.split('/')[1])) ||
           mimeType.toLowerCase().startsWith('video/');
}

function isAudioExtension(fileName) {
    if (!fileName) return false;
    const ext = '.' + fileName.split('.').pop().toLowerCase();
    return CONFIG.SUPPORTED_AUDIO_EXTENSIONS.includes(ext);
}

function isVideoExtension(fileName) {
    if (!fileName) return false;
    const ext = '.' + fileName.split('.').pop().toLowerCase();
    return CONFIG.SUPPORTED_VIDEO_EXTENSIONS.includes(ext);
}

function isPdfFile(mimeType, fileName) {
    if (mimeType === 'application/pdf') return true;
    if (fileName && fileName.toLowerCase().endsWith('.pdf')) return true;
    return false;
}

function getFileType(mimeType, fileName) {
    if (isPdfFile(mimeType, fileName)) return 'pdf';
    if (isAudioMime(mimeType) || isAudioExtension(fileName)) return 'audio';
    if (isVideoMime(mimeType) || isVideoExtension(fileName)) return 'video';
    return 'unknown';
}

function isQuestion(text) {
    if (!text) return false;
    const lowerText = text.toLowerCase().trim();
    if (lowerText.endsWith('?')) return true;
    const questionStarters = [
        'what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom',
        'is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ', 'will ', 'would ',
        'can ', 'could ', 'should ', 'shall ', 'may ', 'might ', 'have ', 'has ',
        'explain', 'tell me', 'describe', 'clarify', 'elaborate', 'meaning of',
        'what\'s', 'what is', 'what are', 'what does', 'what do',
        'is this', 'is it', 'is there', 'are there', 'does this', 'does it'
    ];
    for (const starter of questionStarters) {
        if (lowerText.startsWith(starter)) return true;
    }
    const questionPhrases = [
        'what does', 'what is', 'what are', 'can you explain', 'could you explain',
        'please explain', 'i don\'t understand', 'what about', 'how about',
        'is it', 'are they', 'does it mean', 'does this mean', 'mean by',
        'significance of', 'implications of', 'serious', 'normal', 'abnormal'
    ];
    for (const phrase of questionPhrases) {
        if (lowerText.includes(phrase)) return true;
    }
    return false;
}

const sessionSchema = new mongoose.Schema({
    key: { type: String, required: true, unique: true },
    value: { type: mongoose.Schema.Types.Mixed, required: true },
    updatedAt: { type: Date, default: Date.now }
}, { collection: 'whatsapp_sessions' });

sessionSchema.index({ updatedAt: 1 }, { expireAfterSeconds: 86400 * 30 });

let SessionModel;

async function useMongoDBAuthState() {
    if (!SessionModel) {
        SessionModel = mongoose.model('Session', sessionSchema);
    }

    const writeData = async (key, data) => {
        try {
            const serialized = JSON.stringify(data, (k, v) => {
                if (typeof v === 'bigint') return { type: 'BigInt', value: v.toString() };
                if (v instanceof Uint8Array) return { type: 'Uint8Array', value: Array.from(v) };
                if (Buffer.isBuffer(v)) return { type: 'Buffer', value: Array.from(v) };
                return v;
            });
            
            await SessionModel.findOneAndUpdate(
                { key },
                { key, value: serialized, updatedAt: new Date() },
                { upsert: true, new: true }
            );
        } catch (error) {
            log('❌', `MongoDB write error: ${error.message}`);
        }
    };

    const readData = async (key) => {
        try {
            const doc = await SessionModel.findOne({ key });
            if (!doc || !doc.value) return null;
            
            return JSON.parse(doc.value, (k, v) => {
                if (v && typeof v === 'object') {
                    if (v.type === 'BigInt') return BigInt(v.value);
                    if (v.type === 'Uint8Array') return new Uint8Array(v.data || v.value);
                    if (v.type === 'Buffer') return Buffer.from(v.data || v.value);
                }
                return v;
            });
        } catch (error) {
            log('❌', `MongoDB read error: ${error.message}`);
            return null;
        }
    };

    const removeData = async (key) => {
        try {
            await SessionModel.deleteOne({ key });
        } catch (error) {
            log('❌', `MongoDB delete error: ${error.message}`);
        }
    };

    const clearAll = async () => {
        try {
            await SessionModel.deleteMany({});
            log('🗑️', 'Cleared all MongoDB sessions');
        } catch (error) {
            log('❌', `MongoDB clear error: ${error.message}`);
        }
    };

    let creds = await readData('auth_creds');
    
    if (!creds) {
        const { initAuthCreds } = await import('@whiskeysockets/baileys');
        creds = initAuthCreds();
        await writeData('auth_creds', creds);
        log('🔑', 'Created new auth credentials');
    } else {
        log('🔑', 'Loaded existing auth credentials from MongoDB');
    }

    return {
        state: {
            creds,
            keys: {
                get: async (type, ids) => {
                    const data = {};
                    for (const id of ids) {
                        const value = await readData(`key_${type}_${id}`);
                        if (value) {
                            data[id] = value;
                        }
                    }
                    return data;
                },
                set: async (data) => {
                    const tasks = [];
                    for (const [type, entries] of Object.entries(data)) {
                        for (const [id, value] of Object.entries(entries)) {
                            const key = `key_${type}_${id}`;
                            if (value) {
                                tasks.push(writeData(key, value));
                            } else {
                                tasks.push(removeData(key));
                            }
                        }
                    }
                    await Promise.all(tasks);
                }
            }
        },
        saveCreds: async () => {
            await writeData('auth_creds', creds);
            log('💾', 'Credentials saved to MongoDB');
        },
        clearAll
    };
}

const chatMediaBuffers = new Map();
const chatTimeouts = new Map();
const discoveredGroups = new Map();
const chatContexts = new Map();
const botMessageIds = new Map();

let sock = null;
let isConnected = false;
let qrCodeDataURL = null;
let processedCount = 0;
let botStatus = 'Starting...';
let lastError = null;
let mongoConnected = false;
let authState = null;

let makeWASocket, DisconnectReason, downloadMediaMessage, fetchLatestBaileysVersion;

function log(emoji, message) {
    const time = new Date().toLocaleTimeString();
    console.log(`[${time}] ${emoji} ${message}`);
}

function getSenderId(msg) {
    return msg.key.participant || msg.key.remoteJid;
}

function getSenderName(msg) {
    const senderId = getSenderId(msg);
    const phone = senderId.split('@')[0];
    return msg.pushName || phone;
}

function getShortSenderId(senderId) {
    const phone = senderId.split('@')[0];
    if (phone.length > 6) {
        return phone.slice(-4);
    }
    return phone;
}

function isAllowedGroup(chatId) {
    if (!CONFIG.ALLOWED_GROUP_ID) return false;
    return chatId === CONFIG.ALLOWED_GROUP_ID;
}

function isCommand(text) {
    const lowerText = text.toLowerCase().trim();
    return CONFIG.COMMANDS.includes(lowerText);
}

function getUserBuffer(chatId, senderId) {
    if (!chatMediaBuffers.has(chatId)) {
        chatMediaBuffers.set(chatId, new Map());
    }
    const chatBuffer = chatMediaBuffers.get(chatId);
    if (!chatBuffer.has(senderId)) {
        chatBuffer.set(senderId, []);
    }
    return chatBuffer.get(senderId);
}

function addToUserBuffer(chatId, senderId, mediaItem) {
    const buffer = getUserBuffer(chatId, senderId);
    buffer.push(mediaItem);
    return buffer.length;
}

function clearUserBuffer(chatId, senderId) {
    if (chatMediaBuffers.has(chatId)) {
        const chatBuffer = chatMediaBuffers.get(chatId);
        if (chatBuffer.has(senderId)) {
            const items = chatBuffer.get(senderId);
            chatBuffer.delete(senderId);
            return items;
        }
    }
    return [];
}

function getUserBufferCount(chatId, senderId) {
    if (!chatMediaBuffers.has(chatId)) return 0;
    const chatBuffer = chatMediaBuffers.get(chatId);
    if (!chatBuffer.has(senderId)) return 0;
    return chatBuffer.get(senderId).length;
}

function getTotalBufferStats(chatId) {
    const stats = { users: 0, images: 0, pdfs: 0, audio: 0, video: 0, texts: 0, total: 0 };
    if (!chatMediaBuffers.has(chatId)) return stats;
    
    const chatBuffer = chatMediaBuffers.get(chatId);
    stats.users = chatBuffer.size;
    
    for (const [senderId, items] of chatBuffer) {
        items.forEach(m => {
            if (m.type === 'image') stats.images++;
            else if (m.type === 'pdf') stats.pdfs++;
            else if (m.type === 'audio' || m.type === 'voice') stats.audio++;
            else if (m.type === 'video') stats.video++;
            else if (m.type === 'text') stats.texts++;
            stats.total++;
        });
    }
    return stats;
}

function resetUserTimeout(chatId, senderId, senderName) {
    if (!chatTimeouts.has(chatId)) {
        chatTimeouts.set(chatId, new Map());
    }
    const chatTimeoutMap = chatTimeouts.get(chatId);
    
    if (chatTimeoutMap.has(senderId)) {
        clearTimeout(chatTimeoutMap.get(senderId));
    }
    
    const shortId = getShortSenderId(senderId);
    chatTimeoutMap.set(senderId, setTimeout(() => {
        const clearedItems = clearUserBuffer(chatId, senderId);
        if (clearedItems.length > 0) {
            log('⏰', `Auto-cleared ${clearedItems.length} item(s) for user ...${shortId} after timeout`);
        }
        chatTimeoutMap.delete(senderId);
    }, CONFIG.MEDIA_TIMEOUT_MS));
}

function clearUserTimeout(chatId, senderId) {
    if (chatTimeouts.has(chatId)) {
        const chatTimeoutMap = chatTimeouts.get(chatId);
        if (chatTimeoutMap.has(senderId)) {
            clearTimeout(chatTimeoutMap.get(senderId));
            chatTimeoutMap.delete(senderId);
        }
    }
}

function storeContext(chatId, messageId, mediaFiles, response, senderId) {
    if (!chatContexts.has(chatId)) {
        chatContexts.set(chatId, new Map());
    }
    
    const contexts = chatContexts.get(chatId);
    
    if (contexts.size >= CONFIG.MAX_STORED_CONTEXTS) {
        const entries = Array.from(contexts.entries());
        entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
        const toRemove = entries.slice(0, entries.length - CONFIG.MAX_STORED_CONTEXTS + 1);
        toRemove.forEach(([key]) => contexts.delete(key));
    }
    
    contexts.set(messageId, {
        mediaFiles: mediaFiles,
        response: response,
        timestamp: Date.now(),
        senderId: senderId
    });
    
    log('💾', `Stored context for message ${messageId.substring(0, 8)}...`);
    
    setTimeout(() => {
        if (chatContexts.has(chatId)) {
            const ctx = chatContexts.get(chatId);
            if (ctx.has(messageId)) {
                ctx.delete(messageId);
                log('🧹', `Cleaned up old context ${messageId.substring(0, 8)}...`);
            }
        }
    }, CONFIG.CONTEXT_RETENTION_MS);
}

function getStoredContext(chatId, messageId) {
    if (!chatContexts.has(chatId)) return null;
    const contexts = chatContexts.get(chatId);
    if (!contexts.has(messageId)) return null;
    
    const context = contexts.get(messageId);
    
    if (Date.now() - context.timestamp > CONFIG.CONTEXT_RETENTION_MS) {
        contexts.delete(messageId);
        return null;
    }
    
    return context;
}

function trackBotMessage(chatId, messageId) {
    if (!botMessageIds.has(chatId)) {
        botMessageIds.set(chatId, new Set());
    }
    botMessageIds.get(chatId).add(messageId);
    
    const ids = botMessageIds.get(chatId);
    if (ids.size > 100) {
        const arr = Array.from(ids);
        arr.slice(0, 50).forEach(id => ids.delete(id));
    }
}

function isBotMessage(chatId, messageId) {
    if (!botMessageIds.has(chatId)) return false;
    return botMessageIds.get(chatId).has(messageId);
}

const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    const stats = getTotalBufferStats(CONFIG.ALLOWED_GROUP_ID);
    const storedContextsCount = chatContexts.has(CONFIG.ALLOWED_GROUP_ID) 
        ? chatContexts.get(CONFIG.ALLOWED_GROUP_ID).size 
        : 0;
    
    let html = `
    <!DOCTYPE html>
    <html>
    <head>
        <title>WhatsApp Patient Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="5">
        <style>
            * { box-sizing: border-box; }
            body { font-family: -apple-system, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: linear-gradient(135deg, #25D366 0%, #128C7E 100%); color: white; padding: 20px; }
            .container { text-align: center; background: rgba(255,255,255,0.15); padding: 30px; border-radius: 20px; backdrop-filter: blur(10px); max-width: 600px; width: 100%; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
            h1 { margin: 0 0 10px 0; font-size: 24px; }
            .status { padding: 15px 20px; border-radius: 12px; margin: 20px 0; font-size: 16px; font-weight: 600; }
            .connected { background: #4CAF50; }
            .waiting { background: rgba(255,255,255,0.2); }
            .qr-container { background: white; padding: 15px; border-radius: 15px; display: inline-block; margin: 20px 0; }
            .qr-container img { display: block; max-width: 250px; width: 100%; }
            .stats { display: flex; justify-content: center; gap: 6px; margin-top: 20px; flex-wrap: wrap; }
            .stat { background: rgba(255,255,255,0.1); padding: 8px 10px; border-radius: 10px; min-width: 50px; }
            .stat-value { font-size: 16px; font-weight: bold; }
            .stat-label { font-size: 8px; opacity: 0.8; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 WhatsApp Patient Bot</h1>
            <p>Smart Video & Medical Profile Generator</p>
            <div>ℹ️ API Keys Loaded: ${CONFIG.API_KEYS.length}</div>
    `;
    
    if (isConnected) {
        html += `<div class="status connected">✅ ACTIVE</div>`;
        if (CONFIG.ALLOWED_GROUP_ID) {
            html += `
                <div class="stats">
                    <div class="stat"><div class="stat-value">${stats.users}</div><div class="stat-label">👥 Users</div></div>
                    <div class="stat"><div class="stat-value">${stats.video}</div><div class="stat-label">🎬 Video</div></div>
                    <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">✅ Done</div></div>
                </div>`;
        }
    } else if (qrCodeDataURL) {
        html += `<div class="status waiting">📲 SCAN QR CODE</div><div class="qr-container"><img src="${qrCodeDataURL}" alt="QR Code"></div>`;
    } else {
        html += `<div class="status waiting">⏳ ${botStatus.toUpperCase()}</div>`;
    }
    
    html += `</div></body></html>`;
    res.send(html);
});

app.get('/health', (req, res) => {
    res.json({ status: 'running', connected: isConnected, processedCount });
});

app.listen(PORT, () => {
    log('🌐', `Web server running on port ${PORT}`);
});

async function loadBaileys() {
    botStatus = 'Loading WhatsApp library...';
    try {
        const baileys = await import('@whiskeysockets/baileys');
        makeWASocket = baileys.default || baileys.makeWASocket;
        DisconnectReason = baileys.DisconnectReason;
        downloadMediaMessage = baileys.downloadMediaMessage;
        fetchLatestBaileysVersion = baileys.fetchLatestBaileysVersion;
        return true;
    } catch (error) {
        throw error;
    }
}

async function connectMongoDB() {
    if (!CONFIG.MONGODB_URI) {
        log('⚠️', 'No MONGODB_URI configured - sessions will not persist!');
        return false;
    }
    try {
        await mongoose.connect(CONFIG.MONGODB_URI, {
            serverSelectionTimeoutMS: 10000,
            socketTimeoutMS: 45000,
        });
        mongoConnected = true;
        log('✅', 'MongoDB connected! Sessions will persist.');
        return true;
    } catch (error) {
        mongoConnected = false;
        return false;
    }
}

async function startBot() {
    try {
        botStatus = 'Initializing...';
        log('🚀', 'Starting WhatsApp Bot...');
        
        if (!makeWASocket) await loadBaileys();
        
        let state, saveCreds, clearAll;
        
        if (mongoConnected) {
            const mongoAuth = await useMongoDBAuthState();
            state = mongoAuth.state;
            saveCreds = mongoAuth.saveCreds;
            clearAll = mongoAuth.clearAll;
        } else {
            const { useMultiFileAuthState } = await import('@whiskeysockets/baileys');
            const authPath = join(__dirname, 'auth_session');
            const fileAuth = await useMultiFileAuthState(authPath);
            state = fileAuth.state;
            saveCreds = fileAuth.saveCreds;
            clearAll = async () => { try { fs.rmSync(authPath, { recursive: true, force: true }); } catch(e) {} };
        }
        
        authState = { state, saveCreds, clearAll };
        
        let version;
        try {
            const v = await fetchLatestBaileysVersion();
            version = v.version;
        } catch (e) {
            version = [2, 3000, 1015901307];
        }
        
        botStatus = 'Connecting...';
        
        sock = makeWASocket({
            version,
            auth: state,
            logger: pino({ level: 'silent' }),
            browser: ['WhatsApp-Bot', 'Chrome', '120.0.0'],
            markOnlineOnConnect: false,
            syncFullHistory: false
        });

        sock.ev.on('connection.update', async (update) => {
            const { connection, lastDisconnect, qr } = update;
            
            if (qr) {
                botStatus = 'QR Code ready';
                qrCodeDataURL = await QRCode.toDataURL(qr, { width: 300, margin: 2, color: { dark: '#128C7E', light: '#FFFFFF' } });
                isConnected = false;
            }
            
            if (connection === 'close') {
                isConnected = false;
                qrCodeDataURL = null;
                const statusCode = lastDisconnect?.error?.output?.statusCode;
                
                if (statusCode === DisconnectReason.loggedOut || statusCode === 401 || statusCode === 405) {
                    if (authState?.clearAll) await authState.clearAll();
                    setTimeout(startBot, 5000);
                } else {
                    setTimeout(startBot, 5000);
                }
                
            } else if (connection === 'open') {
                isConnected = true;
                qrCodeDataURL = null;
                botStatus = 'Connected';
                log('✅', '🎉 CONNECTED TO WHATSAPP!');
                
                if (authState?.saveCreds) await authState.saveCreds();
                
                if (CONFIG.ALLOWED_GROUP_ID) {
                    log('🎯', `Bot active ONLY in: ${CONFIG.ALLOWED_GROUP_ID}`);
                }
            }
        });

        sock.ev.on('creds.update', async () => {
            if (authState?.saveCreds) await authState.saveCreds();
        });

        sock.ev.on('messages.upsert', async ({ messages, type }) => {
            if (type !== 'notify') return;
            for (const msg of messages) {
                if (msg.key.fromMe) continue;
                if (!msg.message) continue;
                try {
                    await handleMessage(sock, msg);
                } catch (error) {
                    log('❌', `Message handling error: ${error.message}`);
                }
            }
        });
        
    } catch (error) {
        setTimeout(startBot, 10000);
    }
}

async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    if (chatId === 'status@broadcast') return;
    
    const isGroup = chatId.endsWith('@g.us');
    const senderId = getSenderId(msg);
    const senderName = getSenderName(msg);
    const shortId = getShortSenderId(senderId);
    
    if (isGroup && !discoveredGroups.has(chatId)) {
        try {
            const metadata = await sock.groupMetadata(chatId);
            discoveredGroups.set(chatId, metadata.subject);
            log('📋', `Discovered Group ID: ${chatId}`);
        } catch (e) {
            discoveredGroups.set(chatId, 'Unknown Group');
        }
    }
    
    if (!isAllowedGroup(chatId)) return;
    
    const messageType = Object.keys(msg.message)[0];
    
    let quotedMessageId = null;
    let contextInfo = null;
    
    if (messageType === 'extendedTextMessage') contextInfo = msg.message.extendedTextMessage?.contextInfo;
    else if (messageType === 'imageMessage') contextInfo = msg.message.imageMessage?.contextInfo;
    else if (messageType === 'videoMessage') contextInfo = msg.message.videoMessage?.contextInfo;
    
    if (contextInfo?.stanzaId) {
        quotedMessageId = contextInfo.stanzaId;
        if (isBotMessage(chatId, quotedMessageId)) {
            await handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName, messageType);
            return;
        }
    }
    
    // --- HANDLING DIFFERENT MESSAGE TYPES ---
    
    if (messageType === 'imageMessage') {
        log('📷', `Image from ${senderName} (...${shortId})`);
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            addToUserBuffer(chatId, senderId, {
                type: 'image',
                data: buffer.toString('base64'),
                mimeType: msg.message.imageMessage.mimetype || 'image/jpeg',
                caption: msg.message.imageMessage.caption || '',
                timestamp: Date.now()
            });
            resetUserTimeout(chatId, senderId, senderName);
        } catch (error) { log('❌', `Image error: ${error.message}`); }
    }
    else if (messageType === 'videoMessage') {
        log('🎬', `Video from ${senderName} (...${shortId})`);
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const duration = msg.message.videoMessage.seconds || 0;
            
            addToUserBuffer(chatId, senderId, {
                type: 'video',
                data: buffer.toString('base64'), // Store base64 for now
                mimeType: msg.message.videoMessage.mimetype || 'video/mp4',
                caption: msg.message.videoMessage.caption || '',
                duration: duration,
                timestamp: Date.now()
            });
            
            log('📦', `Video buffered. Duration: ${duration}s`);
            resetUserTimeout(chatId, senderId, senderName);
        } catch (error) { log('❌', `Video error: ${error.message}`); }
    }
    else if (messageType === 'audioMessage') {
        const isVoice = msg.message.audioMessage.ptt === true;
        log(isVoice ? '🎤' : '🎵', `Audio from ${senderName}`);
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            addToUserBuffer(chatId, senderId, {
                type: isVoice ? 'voice' : 'audio',
                data: buffer.toString('base64'),
                mimeType: msg.message.audioMessage.mimetype || 'audio/ogg',
                timestamp: Date.now()
            });
            resetUserTimeout(chatId, senderId, senderName);
        } catch (error) {}
    }
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        const fileType = getFileType(docMime, fileName);
        
        if (fileType !== 'unknown') {
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                addToUserBuffer(chatId, senderId, {
                    type: fileType,
                    data: buffer.toString('base64'),
                    mimeType: docMime,
                    fileName: fileName,
                    timestamp: Date.now()
                });
                log('📎', `Document (${fileType}) buffered`);
                resetUserTimeout(chatId, senderId, senderName);
            } catch (e) {}
        }
    }
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = (msg.message.conversation || msg.message.extendedTextMessage?.text || '').trim();
        if (!text) return;
        
        if (text === CONFIG.TRIGGER_TEXT) {
            log('🔔', `Trigger from ${senderName}`);
            await new Promise(r => setTimeout(r, 1000));
            const userBufferCount = getUserBufferCount(chatId, senderId);
            
            if (userBufferCount > 0) {
                clearUserTimeout(chatId, senderId);
                const mediaFiles = clearUserBuffer(chatId, senderId);
                await processMedia(sock, chatId, mediaFiles, false, null, senderId, senderName, null);
            } else {
                await sock.sendMessage(chatId, { text: `ℹ️ Please send files first, then send *.*` });
            }
        }
        else if (text.toLowerCase() === 'clear') {
            clearUserBuffer(chatId, senderId);
            clearUserTimeout(chatId, senderId);
            await sock.sendMessage(chatId, { text: `🗑️ Buffer cleared.` });
        }
        else if (text.toLowerCase() === 'status') {
             await sock.sendMessage(chatId, { text: `✅ Bot is running. Active keys: ${CONFIG.API_KEYS.length}` });
        }
        else if (!isCommand(text)) {
            addToUserBuffer(chatId, senderId, {
                type: 'text',
                content: text,
                sender: senderName,
                timestamp: Date.now()
            });
            resetUserTimeout(chatId, senderId, senderName);
        }
    }
}

async function handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName, messageType) {
    const storedContext = getStoredContext(chatId, quotedMessageId);
    if (!storedContext) return;
    
    let text = '';
    if (messageType === 'conversation') text = msg.message.conversation;
    else if (messageType === 'extendedTextMessage') text = msg.message.extendedTextMessage?.text;
    
    if (text) {
        const newContent = [{
            type: 'text',
            content: text,
            sender: senderName,
            isFollowUp: true
        }];
        
        const combinedMedia = [...storedContext.mediaFiles, ...newContent];
        await processMedia(sock, chatId, combinedMedia, true, storedContext.response, senderId, senderName, text);
    }
}

async function processMedia(sock, chatId, mediaFiles, isFollowUp = false, previousResponse = null, senderId, senderName, userTextInput = null) {
    const shortId = getShortSenderId(senderId);
    
    try {
        await sock.sendMessage(chatId, { text: `⏳ Processing... (Analysing video/images)` }, { quoted: { key: { id: isFollowUp ? null : 'status-msg' } } });

        const contentParts = [];
        const textNotes = [];
        
        for (const media of mediaFiles) {
            if (media.type === 'text') {
                textNotes.push(media.content);
            } 
            else if (media.type === 'image' || media.type === 'pdf') {
                contentParts.push({
                    inlineData: { data: media.data, mimeType: media.mimeType }
                });
                if (media.caption) textNotes.push(`[Caption]: ${media.caption}`);
            }
            else if (media.type === 'audio' || media.type === 'voice') {
                contentParts.push({
                    inlineData: { data: media.data, mimeType: media.mimeType }
                });
            }
            else if (media.type === 'video') {
                // SMART VIDEO PROCESSING START
                if (media.duration > 0 && media.duration < 120) {
                    // It's a short video (under 2 mins), break it into frames
                    log('🎬', `Smart processing video for ...${shortId}`);
                    const result = await processSmartVideo(media.data, media.mimeType);
                    
                    if (result.success) {
                        log('✅', `Extracted ${result.count} frames from video`);
                        // Add audio and frames to payload
                        result.parts.forEach(p => contentParts.push(p));
                        textNotes.push(`[System Note]: The user provided a video. I have broken it down into ${result.count} frames (3 images per second) to capture fast page flipping. Read these sequential images as a video stream.`);
                        if (media.caption) textNotes.push(`[Video Caption]: ${media.caption}`);
                    } else {
                        // Fallback to standard video
                        contentParts.push({
                            inlineData: { data: media.data, mimeType: media.mimeType }
                        });
                    }
                } else {
                    // Long video, use standard processing (1 FPS limited by Gemini)
                    contentParts.push({
                        inlineData: { data: media.data, mimeType: media.mimeType }
                    });
                }
                // SMART VIDEO PROCESSING END
            }
        }
        
        // Construct Prompt
        let promptText = '';
        if (isFollowUp && previousResponse) {
             const isUserQuestion = userTextInput ? isQuestion(userTextInput) : false;
             if (isUserQuestion) {
                 promptText = `User Question: "${userTextInput}"\n\nPrevious Profile:\n${previousResponse}\n\nAnswer the question based on the medical evidence provided in the images/video frames.`;
             } else {
                 promptText = `Update the Clinical Profile based on this new info: "${userTextInput}"\n\nPrevious Profile:\n${previousResponse}`;
             }
        } else {
             promptText = "Analyze these medical files (extracted frames from video or images). " + (textNotes.length > 0 ? "Additional context: " + textNotes.join('\n') : "");
        }
        
        const requestContent = [promptText, ...contentParts];
        
        // API Call
        let responseText = null;
        const keys = CONFIG.API_KEYS;
        
        for (let i = 0; i < keys.length; i++) {
            try {
                const genAI = new GoogleGenerativeAI(keys[i]);
                const model = genAI.getGenerativeModel({ 
                    model: CONFIG.GEMINI_MODEL,
                    systemInstruction: CONFIG.SYSTEM_INSTRUCTION
                });
                
                const result = await model.generateContent(requestContent);
                responseText = result.response.text();
                break;
            } catch (error) {
                log('⚠️', `Key ${i} failed: ${error.message}`);
                if (i === keys.length - 1) throw error;
            }
        }

        if (!responseText) throw new Error("Empty response");

        const sentMessage = await sock.sendMessage(chatId, { 
            text: responseText,
            mentions: [senderId]
        });
        
        if (sentMessage?.key?.id) {
            trackBotMessage(chatId, sentMessage.key.id);
            storeContext(chatId, sentMessage.key.id, mediaFiles, responseText, senderId);
        }
        
    } catch (error) {
        log('❌', `Error: ${error.message}`);
        await sock.sendMessage(chatId, { text: `❌ Error: ${error.message}` });
    }
}

// Initial Startup
if (CONFIG.API_KEYS.length === 0) log('❌', 'No API Keys found!');
if (!CONFIG.ALLOWED_GROUP_ID) log('⚠️', 'No group configured!');

(async () => {
    try {
        await connectMongoDB();
        await startBot();
    } catch (error) {
        console.error(error);
    }
})();
