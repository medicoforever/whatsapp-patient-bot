import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import pino from 'pino';
import QRCode from 'qrcode';
import express from 'express';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import mongoose from 'mongoose';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import os from 'os';

// Setup FFmpeg path automatically for Render
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Helper to parse keys from comma-separated string
const getApiKeys = () => {
    const keys = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
    return keys.split(',').map(k => k.trim()).filter(k => k.length > 0);
};

// ==============================================================================
// 🟢 NEW CONFIGURATION AREA
// ==============================================================================

const SECONDARY_SYSTEM_INSTRUCTION = `You are an expert radiologist. When you receive a context, it is mostly about a patient and sometimes they might have been advised with any imaging modality. You analyse that info and then advise regarding that as an expert radiologist what to be seen in that specific imaging modality for that specific patient including various hypothetical imaging findings from common to less common for that patient condition in that specific imaging modality. suppose of you cant indentify thr specific imaging modality in thr given context, you yourself choose the appropriate imaging modality based on the specific conditions context`;

const SECONDARY_TRIGGER_PROMPT = `Here is the Clinical Profile generated from the patient's reports. Please analyze this profile according to your system instructions and provide the final output.`;

// ==============================================================================

const CONFIG = {
    // We now store an array of keys
    API_KEYS: getApiKeys(),
    // 🔴 CHANGED TO STABLE MODEL to prevent 503 Overloaded errors
    GEMINI_MODEL: 'gemini-3-flash-preview', 
    MONGODB_URI: process.env.MONGODB_URI,
    
    // Group Routing Configuration
    GROUPS: {
        CT_SOURCE: process.env.GROUP_CT_SOURCE,
        CT_TARGET: process.env.GROUP_CT_TARGET,
        MRI_SOURCE: process.env.GROUP_MRI_SOURCE,
        MRI_TARGET: process.env.GROUP_MRI_TARGET
    },

    MEDIA_TIMEOUT_MS: 300000, // 5 minutes (Standard users)
    AUTO_PROCESS_DELAY_MS: 60000, // 60 seconds (Auto-groups)
    
    CONTEXT_RETENTION_MS: 1800000,
    MAX_STORED_CONTEXTS: 20,
    COMMANDS: ['.', '.1', '.2', '.3', '..', '..1', '..2', '..3', 'help', '?', 'clear', 'status'],
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

// ==============================================================================
// 🔄 API KEY ROTATION LOGIC (Every 2 Hours)
// ==============================================================================
function rotateApiKeys() {
    if (CONFIG.API_KEYS.length > 1) {
        // Remove first element and add to end (Shift Left / Rotate)
        // 1,2,3,4 -> 2,3,4,1
        const key = CONFIG.API_KEYS.shift();
        CONFIG.API_KEYS.push(key);
        log('🔄', `API Keys Rotated. New primary key starts with: ...${CONFIG.API_KEYS[0].slice(-4)}`);
    }
}

// Start rotation interval (2 hours = 7200000 ms)
setInterval(rotateApiKeys, 2 * 60 * 60 * 1000);
// ==============================================================================

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
    
    // Check if ends with question mark
    if (lowerText.endsWith('?')) return true;
    
    // Check for question words at the start
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
    
    // Check for question phrases anywhere
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

// ==============================================================================
// 🧠 HELPER: Smart Grouping by Caption
// ==============================================================================
function groupMediaSmartly(mediaFiles) {
    const distinctCaptions = new Set();
    mediaFiles.forEach(f => {
        if (f.caption && f.caption.trim()) {
            distinctCaptions.add(f.caption.trim());
        }
    });

    // Case 1: Only 1 unique caption (or 0) across all files
    // The user requirement: "if just one caption only found with many images, then that one caption is for all those images as separate one process"
    if (distinctCaptions.size <= 1) {
        return [mediaFiles]; // Return as single batch
    }

    // Case 2: Multiple distinct captions
    // User requirement: "till the beginning of the different caption, those all images belong to that caption which is before that"
    const batches = [];
    let currentBatch = [];
    let activeCaption = null;

    for (const file of mediaFiles) {
        const fileCaption = (file.caption || '').trim();

        // If this file has a caption AND it is different from the current active one
        // It signals the start of a new patient/context
        if (fileCaption && fileCaption !== activeCaption) {
            // Close previous batch if it exists
            if (currentBatch.length > 0) {
                batches.push(currentBatch);
            }
            // Start new batch
            currentBatch = [file];
            activeCaption = fileCaption;
        } else {
            // Append to current batch (same caption OR no caption inheriting previous)
            currentBatch.push(file);
            
            // If this is the very first file and has a caption, set activeCaption
            if (!activeCaption && fileCaption) {
                activeCaption = fileCaption;
            }
        }
    }

    if (currentBatch.length > 0) {
        batches.push(currentBatch);
    }

    return batches;
}

// ==============================================================================
// 🔄 UPDATED TIMEOUT LOGIC (Includes Smart Batching)
// ==============================================================================
function resetUserTimeout(chatId, senderId, senderName) {
    if (!chatTimeouts.has(chatId)) {
        chatTimeouts.set(chatId, new Map());
    }
    const chatTimeoutMap = chatTimeouts.get(chatId);
    
    if (chatTimeoutMap.has(senderId)) {
        clearTimeout(chatTimeoutMap.get(senderId));
    }
    
    // Check if this chat is one of the Auto-Process Source Groups
    const isCTSource = chatId === CONFIG.GROUPS.CT_SOURCE;
    const isMRISource = chatId === CONFIG.GROUPS.MRI_SOURCE;
    const isAutoGroup = isCTSource || isMRISource;

    // Use 60 seconds for auto groups, 5 minutes for others
    const delay = isAutoGroup ? CONFIG.AUTO_PROCESS_DELAY_MS : CONFIG.MEDIA_TIMEOUT_MS;
    
    const shortId = getShortSenderId(senderId);

    const timeoutCallback = async () => {
        if (isAutoGroup) {
            // --- AUTO PROCESSING LOGIC ---
            // If it's a source group, we process automatically and send to target
            const mediaFiles = clearUserBuffer(chatId, senderId);
            if (mediaFiles.length > 0) {
                log('⏱️', `Auto-processing ${mediaFiles.length} item(s) from Source Group (${isCTSource ? 'CT' : 'MRI'})`);
                
                // Determine Target Chat ID
                const targetChatId = isCTSource ? CONFIG.GROUPS.CT_TARGET : CONFIG.GROUPS.MRI_TARGET;
                
                if (targetChatId) {
                    // 🧠 NEW: Smart Grouping Logic
                    const batches = groupMediaSmartly(mediaFiles);
                    
                    if (batches.length > 1) {
                         log('🔀', `Detected ${batches.length} distinct patient contexts. Processing separately.`);
                    }

                    // Process each batch sequentially
                    for (let i = 0; i < batches.length; i++) {
                        const batch = batches[i];
                        if (batch.length === 0) continue;

                        log('▶️', `Processing Batch ${i+1}/${batches.length} (${batch.length} files)`);
                        
                        // Process and send to Target
                        await processMedia(sock, chatId, batch, false, null, senderId, senderName, null, 3, false, targetChatId);
                        
                        // Small delay between batches to ensure order and prevent rate limits
                        if (i < batches.length - 1) {
                            await new Promise(r => setTimeout(r, 2000));
                        }
                    }

                } else {
                    log('⚠️', 'Target group not configured for this source!');
                }
            }
        } else {
            // --- STANDARD BEHAVIOR ---
            // Just clear buffer after long inactivity
            const clearedItems = clearUserBuffer(chatId, senderId);
            if (clearedItems.length > 0) {
                log('⏰', `Auto-cleared ${clearedItems.length} item(s) for user ...${shortId} after timeout`);
            }
        }
        chatTimeoutMap.delete(senderId);
    };

    chatTimeoutMap.set(senderId, setTimeout(timeoutCallback, delay));
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

// === SMART VIDEO PROCESSING LOGIC (Oversample -> Filter) ===
async function extractFramesFromVideo(videoBuffer, targetFps = 3) {
    return new Promise((resolve, reject) => {
        const tempId = Math.random().toString(36).substring(7);
        const tempDir = os.tmpdir();
        const inputPath = join(tempDir, `input_${tempId}.mp4`);
        const outputPattern = join(tempDir, `frame_${tempId}_%03d.jpg`);

        fs.writeFileSync(inputPath, videoBuffer);

        // INTELLIGENT FILTER LOGIC:
        const batchSize = 3;
        const inputFps = targetFps * batchSize;

        const videoFilter = `fps=${inputFps},thumbnail=${batchSize}`;

        log('🎬', `Smart Extract: Target ${targetFps}fps (Input ${inputFps}fps, Batch ${batchSize})`);

        ffmpeg(inputPath)
            .outputOptions([
                `-vf ${videoFilter}`, // The magic intelligent filter
                '-vsync 0',           // Prevent dropping frames arbitrarily
                '-q:v 2'              // High quality JPEG
            ])
            .output(outputPattern)
            .on('end', () => {
                try {
                    const files = fs.readdirSync(tempDir)
                        .filter(f => f.startsWith(`frame_${tempId}_`) && f.endsWith('.jpg'))
                        .sort(); 

                    const frames = files.map(file => {
                        const path = join(tempDir, file);
                        const buffer = fs.readFileSync(path);
                        fs.unlinkSync(path); // Clean up frame
                        return buffer.toString('base64');
                    });

                    fs.unlinkSync(inputPath); // Clean up video
                    resolve(frames);
                } catch (err) {
                    reject(err);
                }
            })
            .on('error', (err) => {
                try { if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath); } catch(e){}
                reject(err);
            })
            .run();
    });
}
// ===================================

const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    // Just grab stats from first available chat buffer for demo, or sum all
    let stats = { users: 0, images: 0, pdfs: 0, audio: 0, video: 0, texts: 0, total: 0 };
    for (const [chatId, _] of chatMediaBuffers) {
         const s = getTotalBufferStats(chatId);
         stats.users += s.users;
         stats.images += s.images;
         stats.pdfs += s.pdfs;
         stats.audio += s.audio;
         stats.video += s.video;
         stats.texts += s.texts;
         stats.total += s.total;
    }
    
    let html = `
    <!DOCTYPE html>
    <html>
    <head>
        <title>WhatsApp Patient Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="5">
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #25D366 0%, #128C7E 100%);
                color: white;
                padding: 20px;
            }
            .container {
                text-align: center;
                background: rgba(255,255,255,0.15);
                padding: 30px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                max-width: 600px;
                width: 100%;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 { margin: 0 0 10px 0; font-size: 24px; }
            .subtitle { opacity: 0.9; margin-bottom: 20px; font-size: 14px; }
            .status {
                padding: 15px 20px;
                border-radius: 12px;
                margin: 20px 0;
                font-size: 16px;
                font-weight: 600;
            }
            .connected { background: #4CAF50; }
            .waiting { background: rgba(255,255,255,0.2); }
            .qr-container {
                background: white;
                padding: 15px;
                border-radius: 15px;
                display: inline-block;
                margin: 20px 0;
            }
            .qr-container img { display: block; max-width: 250px; width: 100%; }
            .info-box {
                text-align: left;
                background: rgba(0,0,0,0.2);
                padding: 15px;
                border-radius: 12px;
                margin-top: 15px;
                font-size: 13px;
            }
            .info-box h3 { margin: 0 0 10px 0; font-size: 15px; }
            .stats {
                display: flex;
                justify-content: center;
                gap: 6px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            .stat {
                background: rgba(255,255,255,0.1);
                padding: 8px 10px;
                border-radius: 10px;
                min-width: 50px;
            }
            .stat-value { font-size: 16px; font-weight: bold; }
            .stat-label { font-size: 8px; opacity: 0.8; }
            .db-status {
                font-size: 11px;
                padding: 5px 10px;
                border-radius: 20px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .db-connected { background: rgba(76, 175, 80, 0.3); }
            .db-disconnected { background: rgba(244, 67, 54, 0.3); }
            .feature-badge {
                display: inline-block;
                background: rgba(255,255,255,0.2);
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 10px;
                margin: 2px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 WhatsApp Patient Bot</h1>
            <p class="subtitle">Medical Clinical Profile Generator</p>
            <div class="db-status ${mongoConnected ? 'db-connected' : 'db-disconnected'}">
                ${mongoConnected ? '🗄️ MongoDB Connected' : '⚠️ MongoDB Not Connected'}
            </div>
            <div>ℹ️ API Keys Loaded: ${CONFIG.API_KEYS.length}</div>
    `;
    
    if (isConnected) {
        html += `
            <div class="status connected">✅ UNIVERSAL MODE ACTIVE</div>
            <p>Bot is active for all incoming messages (Private & Group)</p>
            <div class="stats">
                <div class="stat"><div class="stat-value">${stats.users}</div><div class="stat-label">Active Chats</div></div>
                <div class="stat"><div class="stat-value">${stats.total}</div><div class="stat-label">Buffered</div></div>
                <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">✅ Done</div></div>
            </div>
            <div class="info-box">
                <h3>✨ Features:</h3>
                <p>
                <strong>🌍 Public Access:</strong> Works in any chat/group.<br>
                <strong>🔄 Auto-Groups:</strong> Monitored CT/MRI groups active.<br>
                <strong>🎥 Smart Video:</strong><br>
                - Send <strong>.</strong> for Smart 3 FPS (Best for fast flipping)<br>
                - Send <strong>.2</strong> for Smart 2 FPS<br>
                - Send <strong>.1</strong> for Smart 1 FPS<br>
                <strong>🧠 Secondary Analysis:</strong><br>
                - Send <strong>..</strong> (double dot) for Chained Analysis<br>
                <strong>↩️ Reply:</strong> Reply to bot to ask questions.
                </p>
            </div>
        `;
    } else if (qrCodeDataURL) {
        html += `
            <div class="status waiting">📲 SCAN QR CODE</div>
            <div class="qr-container"><img src="${qrCodeDataURL}" alt="QR Code"></div>
            <div class="info-box">
                <h3>📋 To connect:</h3>
                <p>WhatsApp → ⋮ Menu → Linked Devices → Link a Device</p>
            </div>
        `;
    } else {
        html += `
            <div class="status waiting">⏳ ${botStatus.toUpperCase()}</div>
            <p>Please wait...</p>
        `;
    }
    
    html += `</div></body></html>`;
    res.send(html);
});

app.get('/health', (req, res) => {
    res.json({ 
        status: 'running',
        connected: isConnected,
        mongoConnected: mongoConnected,
        mode: 'universal',
        processedCount,
        activeKeys: CONFIG.API_KEYS.length,
        timestamp: new Date().toISOString()
    });
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
        
        log('✅', 'Baileys loaded!');
        return true;
    } catch (error) {
        log('❌', `Baileys load failed: ${error.message}`);
        throw error;
    }
}

async function connectMongoDB() {
    if (!CONFIG.MONGODB_URI) {
        log('⚠️', 'No MONGODB_URI configured - sessions will not persist!');
        return false;
    }
    
    try {
        log('🔄', 'Connecting to MongoDB...');
        
        mongoose.connection.on('connected', () => {
            log('✅', 'MongoDB connection established');
        });
        
        mongoose.connection.on('error', (err) => {
            log('❌', `MongoDB connection error: ${err.message}`);
        });
        
        mongoose.connection.on('disconnected', () => {
            log('⚠️', 'MongoDB disconnected');
            mongoConnected = false;
        });
        
        await mongoose.connect(CONFIG.MONGODB_URI, {
            serverSelectionTimeoutMS: 10000,
            socketTimeoutMS: 45000,
        });
        
        mongoConnected = true;
        log('✅', 'MongoDB connected! Sessions will persist.');
        return true;
    } catch (error) {
        log('❌', `MongoDB connection failed: ${error.message}`);
        mongoConnected = false;
        return false;
    }
}

async function startBot() {
    try {
        botStatus = 'Initializing...';
        log('🚀', 'Starting WhatsApp Bot...');
        
        if (!makeWASocket) await loadBaileys();

        // 🟢 FIX: Force MongoDB reconnection attempt if configured but disconnected
        if (!mongoConnected && CONFIG.MONGODB_URI) {
            log('⚠️', 'MongoDB appears disconnected. Attempting to reconnect...');
            await connectMongoDB();
        }
        
        let state, saveCreds, clearAll;
        
        if (mongoConnected) {
            try {
                const mongoAuth = await useMongoDBAuthState();
                state = mongoAuth.state;
                saveCreds = mongoAuth.saveCreds;
                clearAll = mongoAuth.clearAll;
                log('✅', 'Using MongoDB for session storage');
            } catch (e) {
                log('❌', `MongoDB auth failed: ${e.message}`);
                throw e;
            }
        } else {
            // Only fall back to file if MongoDB really failed to connect
            const { useMultiFileAuthState } = await import('@whiskeysockets/baileys');
            const authPath = join(__dirname, 'auth_session');
            const fileAuth = await useMultiFileAuthState(authPath);
            state = fileAuth.state;
            saveCreds = fileAuth.saveCreds;
            clearAll = async () => {
                try { fs.rmSync(authPath, { recursive: true, force: true }); } catch(e) {}
            };
            log('📁', 'Using file-based auth (session will be lost on restart)');
        }
        
        authState = { state, saveCreds, clearAll };
        
        let version;
        try {
            const v = await fetchLatestBaileysVersion();
            version = v.version;
            log('📱', `Using WA version: ${version.join('.')}`);
        } catch (e) {
            version = [2, 3000, 1015901307];
            log('⚠️', 'Using fallback WA version');
        }
        
        botStatus = 'Connecting...';
        
        sock = makeWASocket({
            version,
            auth: state,
            logger: pino({ level: 'silent' }),
            browser: ['WhatsApp-Bot', 'Chrome', '120.0.0'],
            markOnlineOnConnect: false,
            syncFullHistory: false,
            getMessage: async (key) => {
                return { conversation: '' };
            }
        });

        sock.ev.on('connection.update', async (update) => {
            const { connection, lastDisconnect, qr } = update;
            
            if (qr) {
                try {
                    botStatus = 'QR Code ready';
                    qrCodeDataURL = await QRCode.toDataURL(qr, {
                        width: 300, 
                        margin: 2,
                        color: { dark: '#128C7E', light: '#FFFFFF' }
                    });
                    isConnected = false;
                    log('📱', 'QR Code generated - please scan!');
                } catch (err) {
                    log('❌', `QR generation error: ${err.message}`);
                    lastError = err.message;
                }
            }
            
            if (connection === 'close') {
                isConnected = false;
                qrCodeDataURL = null;
                
                const statusCode = lastDisconnect?.error?.output?.statusCode;
                const reason = lastDisconnect?.error?.output?.payload?.message || 'Unknown';
                
                log('🔌', `Connection closed. Code: ${statusCode}, Reason: ${reason}`);
                
                const loggedOut = statusCode === DisconnectReason.loggedOut || 
                                  statusCode === 401 || 
                                  statusCode === 405;
                
                if (loggedOut) {
                    log('🔐', 'Session logged out - clearing credentials...');
                    botStatus = 'Logged out - clearing session...';
                    
                    if (authState?.clearAll) {
                        await authState.clearAll();
                    }
                    
                    log('🔄', 'Restarting with fresh session in 5 seconds...');
                    setTimeout(startBot, 5000);
                } else {
                    const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
                    log('🔄', `Reconnecting in 5 seconds... (shouldReconnect: ${shouldReconnect})`);
                    setTimeout(startBot, 5000);
                }
                
            } else if (connection === 'open') {
                isConnected = true;
                qrCodeDataURL = null;
                botStatus = 'Connected';
                
                log('✅', '🎉 CONNECTED TO WHATSAPP!');
                
                if (authState?.saveCreds) {
                    await authState.saveCreds();
                    log('💾', 'Credentials saved');
                }
                
                log('🌍', 'Universal Mode: Bot is active for ALL chats.');
                if (CONFIG.GROUPS.CT_SOURCE) log('🏥', 'Monitoring CT Source Group');
                if (CONFIG.GROUPS.MRI_SOURCE) log('🏥', 'Monitoring MRI Source Group');
            }
        });

        sock.ev.on('creds.update', async () => {
            if (authState?.saveCreds) {
                await authState.saveCreds();
            }
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
        log('💥', `Bot error: ${error.message}`);
        console.error(error);
        botStatus = 'Error - restarting...';
        setTimeout(startBot, 10000);
    }
}

async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    
    if (chatId === 'status@broadcast') return;
    
    const senderId = getSenderId(msg);
    const senderName = getSenderName(msg);
    const shortId = getShortSenderId(senderId);
    
    // Group discovery logging (optional, no longer restricts access)
    const isGroup = chatId.endsWith('@g.us');
    if (isGroup) {
         log('📋', `Message from group: ${chatId} (Allowed: ALL)`);
    }

    // No isAllowedGroup check here -> Public Bot
    
    const messageType = Object.keys(msg.message)[0];
    
    let quotedMessageId = null;
    let contextInfo = null;
    
    if (messageType === 'extendedTextMessage') {
        contextInfo = msg.message.extendedTextMessage?.contextInfo;
    } else if (messageType === 'imageMessage') {
        contextInfo = msg.message.imageMessage?.contextInfo;
    } else if (messageType === 'documentMessage') {
        contextInfo = msg.message.documentMessage?.contextInfo;
    } else if (messageType === 'audioMessage') {
        contextInfo = msg.message.audioMessage?.contextInfo;
    } else if (messageType === 'videoMessage') {
        contextInfo = msg.message.videoMessage?.contextInfo;
    }
    
    if (contextInfo?.stanzaId) {
        quotedMessageId = contextInfo.stanzaId;
        
        if (isBotMessage(chatId, quotedMessageId)) {
            log('↩️', `Reply to bot from ${senderName} (...${shortId})`);
            await handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName, messageType);
            return;
        }
    }
    
    if (messageType === 'imageMessage') {
        log('📷', `Image from ${senderName} (...${shortId})`);
        
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const caption = msg.message.imageMessage.caption || '';
            
            const count = addToUserBuffer(chatId, senderId, {
                type: 'image',
                data: buffer.toString('base64'),
                mimeType: msg.message.imageMessage.mimetype || 'image/jpeg',
                caption: caption,
                timestamp: Date.now()
            });
            
            if (caption) {
                log('💬', `  └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
            }
            
            log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
            resetUserTimeout(chatId, senderId, senderName);
            
        } catch (error) {
            log('❌', `Image error: ${error.message}`);
        }
    }
    else if (messageType === 'videoMessage') {
        log('🎬', `Video from ${senderName} (...${shortId})`);
        
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const caption = msg.message.videoMessage.caption || '';
            const mimeType = msg.message.videoMessage.mimetype || 'video/mp4';
            
            const count = addToUserBuffer(chatId, senderId, {
                type: 'video',
                data: buffer.toString('base64'),
                mimeType: mimeType,
                caption: caption,
                duration: msg.message.videoMessage.seconds || 0,
                timestamp: Date.now()
            });
            
            if (caption) {
                log('💬', `  └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
            }
            
            log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
            resetUserTimeout(chatId, senderId, senderName);
            
        } catch (error) {
            log('❌', `Video error: ${error.message}`);
        }
    }
    else if (messageType === 'audioMessage') {
        const isVoice = msg.message.audioMessage.ptt === true;
        const emoji = isVoice ? '🎤' : '🎵';
        
        log(emoji, `${isVoice ? 'Voice' : 'Audio'} from ${senderName} (...${shortId})`);
        
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            
            const count = addToUserBuffer(chatId, senderId, {
                type: isVoice ? 'voice' : 'audio',
                data: buffer.toString('base64'),
                mimeType: msg.message.audioMessage.mimetype || 'audio/ogg',
                duration: msg.message.audioMessage.seconds || 0,
                timestamp: Date.now()
            });
            
            log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
            resetUserTimeout(chatId, senderId, senderName);
            
        } catch (error) {
            log('❌', `Audio error: ${error.message}`);
        }
    }
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        const caption = msg.message.documentMessage.caption || '';
        
        const fileType = getFileType(docMime, fileName);
        
        if (fileType === 'pdf') {
            log('📄', `PDF from ${senderName} (...${shortId}): ${fileName}`);
            
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                const count = addToUserBuffer(chatId, senderId, {
                    type: 'pdf',
                    data: buffer.toString('base64'),
                    mimeType: 'application/pdf',
                    fileName: fileName,
                    caption: caption,
                    timestamp: Date.now()
                });
                
                if (caption) {
                    log('💬', `  └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
                }
                
                log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
                resetUserTimeout(chatId, senderId, senderName);
                
            } catch (error) {
                log('❌', `PDF error: ${error.message}`);
            }
        }
        else if (fileType === 'audio') {
            log('🎵', `Audio file from ${senderName} (...${shortId}): ${fileName}`);
            
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                const count = addToUserBuffer(chatId, senderId, {
                    type: 'audio',
                    data: buffer.toString('base64'),
                    mimeType: docMime || 'audio/mpeg',
                    fileName: fileName,
                    caption: caption,
                    timestamp: Date.now()
                });
                
                if (caption) {
                    log('💬', `  └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
                }
                
                log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
                resetUserTimeout(chatId, senderId, senderName);
                
            } catch (error) {
                log('❌', `Audio file error: ${error.message}`);
            }
        }
        else if (fileType === 'video') {
            log('🎬', `Video file from ${senderName} (...${shortId}): ${fileName}`);
            
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                const count = addToUserBuffer(chatId, senderId, {
                    type: 'video',
                    data: buffer.toString('base64'),
                    mimeType: docMime || 'video/mp4',
                    fileName: fileName,
                    caption: caption,
                    timestamp: Date.now()
                });
                
                if (caption) {
                    log('💬', `  └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
                }
                
                log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
                resetUserTimeout(chatId, senderId, senderName);
                
            } catch (error) {
                log('❌', `Video file error: ${error.message}`);
            }
        }
        else {
            log('📎', `Skipping unsupported file: ${fileName} (${docMime})`);
        }
    }
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = (msg.message.conversation || msg.message.extendedTextMessage?.text || '').trim();
        
        if (!text) return;

        // CHECK FOR TRIGGERS
        // Primary: . .1 .2 .3
        const isPrimaryTrigger = /^(\.|(\.[1-3]))$/.test(text);
        // Secondary: .. ..1 ..2 ..3
        const isSecondaryTrigger = /^(\.\.|(\.\.[1-3]))$/.test(text);
        
        if (isPrimaryTrigger || isSecondaryTrigger) {
            log('🔔', `Trigger command "${text}" from ${senderName} (...${shortId})`);
            
            await new Promise(r => setTimeout(r, 1000));
            
            const userBufferCount = getUserBufferCount(chatId, senderId);
            
            if (userBufferCount > 0) {
                clearUserTimeout(chatId, senderId);
                const mediaFiles = clearUserBuffer(chatId, senderId);
                
                // DETERMINE VIDEO FPS based on command suffix
                // . or .. = 3fps
                // .1 or ..1 = 1fps
                // .2 or ..2 = 2fps
                const lastChar = text.slice(-1);
                let targetFps = 3; 
                if (!isNaN(parseInt(lastChar))) {
                    targetFps = parseInt(lastChar);
                }
                
                // 🧠 NEW: Smart Grouping Logic for Manual Triggers too
                const batches = groupMediaSmartly(mediaFiles);
                
                if (batches.length > 1) {
                    log('🔀', `Manual Trigger: Detected ${batches.length} distinct patient contexts.`);
                }
                
                // Process each batch sequentially
                for (let i = 0; i < batches.length; i++) {
                    const batch = batches[i];
                    if (batch.length === 0) continue;
                    
                    const modeLabel = isSecondaryTrigger ? 'SECONDARY/CHAINED' : 'PRIMARY';
                    log('🤖', `Processing Batch ${i+1}/${batches.length} (${batch.length} items) with FPS=${targetFps}. Mode: ${modeLabel}`);
                    
                    // Pass the isSecondaryMode flag (true if .. is used)
                    await processMedia(sock, chatId, batch, false, null, senderId, senderName, null, targetFps, isSecondaryTrigger);
                    
                    if (i < batches.length - 1) {
                        await new Promise(r => setTimeout(r, 2000));
                    }
                }

            } else {
                await sock.sendMessage(chatId, { 
                    text: `ℹ️ @${senderId.split('@')[0]}, you have no files buffered.\n\nSend files first, then send *.* (Standard) or *..* (Secondary Analysis).\nAdd numbers for video speed (e.g. .2 or ..2)\n\n💡 _Or reply to my previous response to ask questions!_`,
                    mentions: [senderId]
                });
            }
        }
        else if (text.toLowerCase() === 'help' || text === '?') {
            await sock.sendMessage(chatId, { 
                text: `🏥 *Clinical Profile Bot*\n\n*Universal Mode Active*\nI work in this chat and any group I'm added to!\n\n*Supported Files:*\n📷 Images, 📄 PDFs, 🎤 Voice, 🎵 Audio, 🎬 Video\n\n*Commands:*\n• *.*  - Standard Clinical Profile (Smart 3 FPS)\n• *..* - Secondary Chained Analysis (Profile + Advice)\n• *.1 / ..1* - Process with Smart 1 FPS\n• *.2 / ..2* - Process with Smart 2 FPS\n• *clear* - Clear buffer\n• *status* - Check status\n\n*Reply Feature:*\nReply to my messages to ask questions or provide corrections!` 
            });
        }
        else if (text.toLowerCase() === 'clear') {
            const userItems = clearUserBuffer(chatId, senderId);
            clearUserTimeout(chatId, senderId);
            
            if (userItems.length > 0) {
                const counts = { images: 0, pdfs: 0, audio: 0, video: 0, texts: 0 };
                userItems.forEach(m => {
                    if (m.type === 'image') counts.images++;
                    else if (m.type === 'pdf') counts.pdfs++;
                    else if (m.type === 'audio' || m.type === 'voice') counts.audio++;
                    else if (m.type === 'video') counts.video++;
                    else if (m.type === 'text') counts.texts++;
                });
                
                await sock.sendMessage(chatId, { 
                    text: `🗑️ @${senderId.split('@')[0]}, cleared your buffer:\n📷 ${counts.images} image(s)\n📄 ${counts.pdfs} PDF(s)\n🎵 ${counts.audio} audio\n🎬 ${counts.video} video(s)\n💬 ${counts.texts} text(s)`,
                    mentions: [senderId]
                });
            } else {
                await sock.sendMessage(chatId, { 
                    text: `ℹ️ @${senderId.split('@')[0]}, your buffer is empty.`,
                    mentions: [senderId]
                });
            }
        }
        else if (text.toLowerCase() === 'status') {
            const stats = getTotalBufferStats(chatId);
            const userCount = getUserBufferCount(chatId, senderId);
            const storedContexts = chatContexts.has(chatId) ? chatContexts.get(chatId).size : 0;
            
            await sock.sendMessage(chatId, { 
                text: `📊 *Status*\n\n*Your Buffer:* ${userCount} item(s)\n\n*Chat Total:*\n👥 Active users: ${stats.users}\n📷 Images: ${stats.images}\n📄 PDFs: ${stats.pdfs}\n🎵 Audio: ${stats.audio}\n🎬 Video: ${stats.video}\n💬 Texts: ${stats.texts}\n━━━━━━━━━━\n📦 Total buffered: ${stats.total}\n🧠 Stored contexts: ${storedContexts}\n✅ Processed: ${processedCount}\n🗄️ MongoDB: ${mongoConnected ? 'Connected' : 'Not connected'}\n🔑 API Keys: ${CONFIG.API_KEYS.length} available` 
            });
        }
        else {
            log('💬', `Text from ${senderName} (...${shortId}): "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`);
            
            const count = addToUserBuffer(chatId, senderId, {
                type: 'text',
                content: text,
                sender: senderName,
                timestamp: Date.now()
            });
            
            log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
            resetUserTimeout(chatId, senderId, senderName);
        }
    }
}

async function handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName, messageType) {
    const storedContext = getStoredContext(chatId, quotedMessageId);
    const shortId = getShortSenderId(senderId);
    
    if (!storedContext) {
        log('⚠️', `Context expired for ...${shortId}`);
        await sock.sendMessage(chatId, { 
            text: `⏰ @${senderId.split('@')[0]}, that context has expired (30 min limit).\n\nPlease send new files and use "." to process.`,
            mentions: [senderId]
        });
        return;
    }
    
    const newContent = [];
    let userTextInput = '';
    
    // Handle both conversation and extendedTextMessage for text replies
    if (messageType === 'conversation') {
        const text = msg.message.conversation || '';
        if (text.trim()) {
            userTextInput = text.trim();
            newContent.push({
                type: 'text',
                content: text.trim(),
                sender: senderName,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('💬', `Follow-up text (conversation) from ...${shortId}: "${text.substring(0, 50)}..."`);
        }
    }
    else if (messageType === 'extendedTextMessage') {
        const text = msg.message.extendedTextMessage?.text || '';
        if (text.trim()) {
            userTextInput = text.trim();
            newContent.push({
                type: 'text',
                content: text.trim(),
                sender: senderName,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('💬', `Follow-up text (extended) from ...${shortId}: "${text.substring(0, 50)}..."`);
        }
    }
    else if (messageType === 'imageMessage') {
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const caption = msg.message.imageMessage.caption || '';
            
            newContent.push({
                type: 'image',
                data: buffer.toString('base64'),
                mimeType: msg.message.imageMessage.mimetype || 'image/jpeg',
                caption: caption,
                timestamp: Date.now(),
                isFollowUp: true
            });
            if (caption) userTextInput = caption;
            log('📷', `Follow-up image from ...${shortId}`);
        } catch (error) {
            log('❌', `Image error: ${error.message}`);
        }
    }
    else if (messageType === 'videoMessage') {
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const caption = msg.message.videoMessage.caption || '';
            
            newContent.push({
                type: 'video',
                data: buffer.toString('base64'),
                mimeType: msg.message.videoMessage.mimetype || 'video/mp4',
                caption: caption,
                duration: msg.message.videoMessage.seconds || 0,
                timestamp: Date.now(),
                isFollowUp: true
            });
            if (caption) userTextInput = caption;
            log('🎬', `Follow-up video from ...${shortId}`);
        } catch (error) {
            log('❌', `Video error: ${error.message}`);
        }
    }
    else if (messageType === 'audioMessage') {
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const isVoice = msg.message.audioMessage.ptt === true;
            
            newContent.push({
                type: isVoice ? 'voice' : 'audio',
                data: buffer.toString('base64'),
                mimeType: msg.message.audioMessage.mimetype || 'audio/ogg',
                duration: msg.message.audioMessage.seconds || 0,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log(isVoice ? '🎤' : '🎵', `Follow-up audio from ...${shortId}`);
        } catch (error) {
            log('❌', `Audio error: ${error.message}`);
        }
    }
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        const caption = msg.message.documentMessage.caption || '';
        
        const fileType = getFileType(docMime, fileName);
        
        if (fileType === 'pdf') {
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                newContent.push({
                    type: 'pdf',
                    data: buffer.toString('base64'),
                    mimeType: 'application/pdf',
                    fileName: fileName,
                    caption: caption,
                    timestamp: Date.now(),
                    isFollowUp: true
                });
                if (caption) userTextInput = caption;
                log('📄', `Follow-up PDF from ...${shortId}`);
            } catch (error) {
                log('❌', `PDF error: ${error.message}`);
            }
        }
        else if (fileType === 'audio') {
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                newContent.push({
                    type: 'audio',
                    data: buffer.toString('base64'),
                    mimeType: docMime || 'audio/mpeg',
                    fileName: fileName,
                    caption: caption,
                    timestamp: Date.now(),
                    isFollowUp: true
                });
                if (caption) userTextInput = caption;
                log('🎵', `Follow-up audio file from ...${shortId}`);
            } catch (error) {
                log('❌', `Audio file error: ${error.message}`);
            }
        }
        else if (fileType === 'video') {
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                newContent.push({
                    type: 'video',
                    data: buffer.toString('base64'),
                    mimeType: docMime || 'video/mp4',
                    fileName: fileName,
                    caption: caption,
                    timestamp: Date.now(),
                    isFollowUp: true
                });
                if (caption) userTextInput = caption;
                log('🎬', `Follow-up video file from ...${shortId}`);
            } catch (error) {
                log('❌', `Video file error: ${error.message}`);
            }
        }
    }
    
    if (newContent.length === 0) {
        await sock.sendMessage(chatId, { 
            text: `ℹ️ @${senderId.split('@')[0]}, please include text, image, PDF, audio, or video in your reply.`,
            mentions: [senderId]
        });
        return;
    }
    
    // Determine if this is a question or additional context
    const isUserQuestion = isQuestion(userTextInput);
    
    const combinedMedia = [...storedContext.mediaFiles, ...newContent];
    
    log('🔄', `Regenerating for ...${shortId}: ${storedContext.mediaFiles.length} original + ${newContent.length} new (isQuestion: ${isUserQuestion})`);
    
    await processMedia(sock, chatId, combinedMedia, true, storedContext.response, senderId, senderName, userTextInput);
}

// Helper Function for Gemini API Calls with Rotation
async function generateGeminiContent(requestContent, systemInstruction) {
    const keys = CONFIG.API_KEYS;
    if (keys.length === 0) {
        throw new Error('No API keys configured!');
    }

    let responseText = null;
    let lastErrorMsg = '';

    // 🔴 NEW: Explicit Safety Settings to allow medical content
    const safetySettings = [
        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    ];

    for (let i = 0; i < keys.length; i++) {
        try {
            // 🔴 NEW: Add a 2-second delay before retrying with next key
            if (i > 0) {
                log('⚠️', `Waiting 2s before retrying with Backup Key #${i + 1}...`);
                await new Promise(resolve => setTimeout(resolve, 2000));
            }

            const genAI = new GoogleGenerativeAI(keys[i]);
            const model = genAI.getGenerativeModel({ 
                model: CONFIG.GEMINI_MODEL,
                systemInstruction: systemInstruction,
                safetySettings: safetySettings // 🔴 Apply Safety Settings
            });
            
            const result = await model.generateContent(requestContent);
            responseText = result.response.text();
            
            // 🟢 MODIFICATION START: Force error on empty response to trigger key rotation
            if (!responseText) {
                 const feedback = JSON.stringify(result.response.promptFeedback || {});
                 throw new Error(`Empty response from API (Safety/Filter/Glitch). Feedback: ${feedback}`);
            }
            // 🟢 MODIFICATION END

            return responseText; // Success

        } catch (error) {
            lastErrorMsg = error.message;
            log('❌', `Key #${i + 1} failed: ${error.message}`);
        }
    }
    throw new Error(`All ${keys.length} API keys failed. Last error: ${lastErrorMsg}`);
}


// 🔄 UPDATED processMedia to support Target Chat routing and Caption Headers
// 🟢 MODIFICATION: Added retryAttempt parameter (default 0)
async function processMedia(sock, chatId, mediaFiles, isFollowUp = false, previousResponse = null, senderId, senderName, userTextInput = null, targetFps = 3, isSecondaryMode = false, targetChatId = null, retryAttempt = 0) {
    const shortId = getShortSenderId(senderId);
    // If targetChatId is provided, we send the result there. Otherwise, back to original chatId.
    const destinationChatId = targetChatId || chatId;
    
    try {
        const counts = { images: 0, pdfs: 0, audio: 0, video: 0, texts: 0, followUps: 0 };
        const textContents = [];
        const captions = [];
        const binaryMedia = [];
        const followUpTexts = [];
        
        // === VIDEO PRE-PROCESSING START ===
        const processedMedia = [];
        
        for (const m of mediaFiles) {
            if (m.type === 'video') {
                log('🎬', `Processing video for ...${shortId} (Target FPS: ${targetFps})`);
                try {
                    const videoBuffer = Buffer.from(m.data, 'base64');
                    // Extract frames using ffmpeg with smart filter
                    const frames = await extractFramesFromVideo(videoBuffer, targetFps);
                    log('📸', `Extracted ${frames.length} smart frames from video`);
                    
                    // Add extracted frames as individual images
                    frames.forEach(frameData => {
                        processedMedia.push({
                            type: 'image',
                            data: frameData,
                            mimeType: 'image/jpeg',
                            caption: m.caption ? `[Frame from video] ${m.caption}` : '[Frame from video]',
                            timestamp: m.timestamp,
                            isFollowUp: m.isFollowUp
                        });
                    });
                } catch (err) {
                    log('❌', `Video extraction failed: ${err.message}. processing as standard video.`);
                    // Fallback to original if ffmpeg fails
                    processedMedia.push(m);
                }
            } else {
                processedMedia.push(m);
            }
        }
        // === VIDEO PRE-PROCESSING END ===
        
        processedMedia.forEach(m => {
            if (m.isFollowUp) counts.followUps++;
            
            if (m.type === 'image') {
                counts.images++;
                binaryMedia.push(m);
                if (m.caption) {
                    if (m.isFollowUp) {
                        followUpTexts.push(`[Additional image caption]: ${m.caption}`);
                    } else {
                        captions.push(`[Image caption]: ${m.caption}`);
                    }
                }
            }
            else if (m.type === 'pdf') {
                counts.pdfs++;
                binaryMedia.push(m);
                if (m.caption) {
                    if (m.isFollowUp) {
                        followUpTexts.push(`[Additional PDF caption]: ${m.caption}`);
                    } else {
                        captions.push(`[PDF caption]: ${m.caption}`);
                    }
                }
            }
            else if (m.type === 'audio' || m.type === 'voice') {
                counts.audio++;
                binaryMedia.push(m);
                if (m.caption) {
                    if (m.isFollowUp) {
                        followUpTexts.push(`[Additional audio caption]: ${m.caption}`);
                    } else {
                        captions.push(`[Audio caption]: ${m.caption}`);
                    }
                }
            }
            else if (m.type === 'video') {
                counts.video++;
                binaryMedia.push(m);
                if (m.caption) {
                    if (m.isFollowUp) {
                        followUpTexts.push(`[Additional video caption]: ${m.caption}`);
                    } else {
                        captions.push(`[Video caption]: ${m.caption}`);
                    }
                }
            }
            else if (m.type === 'text') {
                counts.texts++;
                if (m.isFollowUp) {
                    followUpTexts.push(`[User follow-up from ${m.sender}]: ${m.content}`);
                } else {
                    textContents.push(`[Text note from ${m.sender}]: ${m.content}`);
                }
            }
        });
        
        if (isFollowUp) {
            log('🤖', `Processing follow-up for ...${shortId}: ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.video} video, ${counts.texts} text`);
        } else {
            log('🤖', `Processing for ...${shortId}: ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.video} video, ${counts.texts} text`);
        }
        
        const contentParts = [];
        binaryMedia.forEach(media => {
            contentParts.push({
                inlineData: { 
                    data: media.data, 
                    mimeType: media.mimeType 
                }
            });
        });
        
        let promptParts = [];
        if (counts.images > 0) promptParts.push(`${counts.images} image(s)`);
        if (counts.pdfs > 0) promptParts.push(`${counts.pdfs} PDF document(s)`);
        if (counts.audio > 0) promptParts.push(`${counts.audio} audio/voice recording(s)`);
        if (counts.video > 0) promptParts.push(`${counts.video} video file(s)`);
        
        const allOriginalText = [...captions, ...textContents];
        let promptText = '';
        
        if (isFollowUp && previousResponse) {
            // Determine if this is a question or additional context
            const isUserQuestion = userTextInput ? isQuestion(userTextInput) : false;
            
            if (isUserQuestion) {
                // User is asking a question - answer it based on context
                promptText = `The user is replying to a previously generated Clinical Profile with a QUESTION or request for clarification.

=== PREVIOUS CLINICAL PROFILE ===
${previousResponse}
=== END PREVIOUS CLINICAL PROFILE ===

=== ORIGINAL MEDICAL CONTENT CONTEXT ===
${allOriginalText.length > 0 ? allOriginalText.join('\n\n') : '(Original medical files are attached for reference)'}
=== END ORIGINAL CONTEXT ===

=== USER'S QUESTION/REQUEST ===
${followUpTexts.join('\n\n')}
=== END USER'S QUESTION ===

Please answer the user's question directly and helpfully based on the Clinical Profile and original medical content. 
- Provide clear, understandable explanations
- If appropriate, explain medical terms in simple language
- Be informative and thorough
- If they ask about specific findings, explain what those findings typically mean
- Remind them this is AI analysis for informational purposes only and not a substitute for professional medical advice

DO NOT regenerate the Clinical Profile unless specifically asked. Just answer their question.`;
            } else {
                // User is providing additional context or corrections - regenerate profile
                promptText = `The user is replying to a previously generated Clinical Profile with ADDITIONAL CONTEXT, CORRECTIONS, or NEW INFORMATION.

=== PREVIOUS CLINICAL PROFILE ===
${previousResponse}
=== END PREVIOUS CLINICAL PROFILE ===

=== ORIGINAL CONTEXT ===
${allOriginalText.length > 0 ? allOriginalText.join('\n\n') : '(Original files are attached below)'}
=== END ORIGINAL CONTEXT ===

=== NEW ADDITIONAL INFORMATION FROM USER ===
${followUpTexts.join('\n\n')}
=== END NEW INFORMATION ===

Please analyze ALL the content (original files + original text + NEW additional information) and generate an UPDATED Clinical Profile that incorporates the new information. Follow the standard Clinical Profile format:
- Single paragraph starting with "Clinical Profile:"
- Wrapped in single asterisks
- Chronological order for dated scans
- Exclude patient name, age, gender`;
            }
            
        } else if (binaryMedia.length > 0 && allOriginalText.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} along with the following additional text notes/context, and generate the Clinical Profile.

=== ADDITIONAL TEXT NOTES ===
${allOriginalText.join('\n\n')}
=== END OF TEXT NOTES ===

For audio files, transcribe the content first, then extract medical information.
For video files (or video frames provided as images), analyze visual content and extract text.`;
        } 
        else if (binaryMedia.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} containing medical information and generate the Clinical Profile. For audio files, transcribe the content first. For video files (or video frames provided as images), analyze visual content and extract text.`;
        }
        else if (allOriginalText.length > 0) {
            promptText = `Analyze the following text notes containing medical information and generate the Clinical Profile.

=== TEXT NOTES ===
${allOriginalText.join('\n\n')}
=== END OF TEXT NOTES ===`;
        }
        
        let requestContent;
        if (binaryMedia.length > 0) {
            requestContent = [promptText, ...contentParts];
        } else {
            requestContent = [promptText];
        }
        
        // --- STEP 1: Generate Primary Clinical Profile ---
        log('🔄', `Generating Primary Response (Secondary Mode: ${isSecondaryMode})...`);
        const primaryResponseText = await generateGeminiContent(requestContent, CONFIG.SYSTEM_INSTRUCTION);
        
        // If Secondary Mode is active, we need to send the primary response first, then continue
        if (isSecondaryMode && !isFollowUp) {
            await sock.sendMessage(destinationChatId, { 
                text: `📝 *Clinical Profile (Step 1):*\n\n${primaryResponseText}`,
                mentions: [senderId]
            });
            log('📤', `Sent Primary (Step 1) to ...${shortId}`);

            // --- STEP 2: Generate Secondary Analysis ---
            log('🔄', `Generating Secondary Analysis...`);
            
            const secondaryPrompt = `${SECONDARY_TRIGGER_PROMPT}

=== CLINICAL PROFILE ===
${primaryResponseText}
=== END PROFILE ===`;

            const secondaryRequestContent = [secondaryPrompt]; // Text only request based on profile
            const secondaryResponseText = await generateGeminiContent(secondaryRequestContent, SECONDARY_SYSTEM_INSTRUCTION);
            
            // The final response we want to store and send is the Secondary one (or a combo)
            // We'll treat the Secondary response as the "Result" for context purposes.
            const finalSecondaryText = `🧠 *Secondary Analysis (Step 2):*\n\n${secondaryResponseText}`;
            
            // Update responseText variable for the final send block below
            processedCount++;
            
            console.log('\n' + '═'.repeat(60));
            console.log(`👤 User: ${senderName} (...${shortId})`);
            console.log(`📊 CHAINED ANALYSIS COMPLETE`);
            console.log(`⏰ ${new Date().toLocaleString()}`);
            console.log('═'.repeat(60));
            console.log(finalSecondaryText);
            console.log('═'.repeat(60) + '\n');
            
            await sock.sendPresenceUpdate('composing', destinationChatId);
            await new Promise(resolve => setTimeout(resolve, 2000));
            await sock.sendPresenceUpdate('paused', destinationChatId);
            
            const sentMessage = await sock.sendMessage(destinationChatId, { 
                text: finalSecondaryText,
                mentions: [senderId]
            });
            
            if (sentMessage?.key?.id) {
                const messageId = sentMessage.key.id;
                trackBotMessage(destinationChatId, messageId);
                // Store the SECONDARY response as the context context for follow-ups
                // Note: We store context in destination chat so reply works there
                storeContext(destinationChatId, messageId, mediaFiles, secondaryResponseText, senderId);
                log('💾', `Secondary Context stored for ...${shortId}`);
            }
            log('📤', `Sent Secondary (Step 2) to target!`);
            return; // Exit here as we handled sending manually for secondary mode
        }

        // --- NORMAL PRIMARY MODE or FOLLOW-UP HANDLING ---
        
        if (!primaryResponseText || primaryResponseText.trim() === '') {
            // This should be caught by generateGeminiContent now, but safe fallback
            throw new Error('Received empty response from AI');
        }
        
        log('✅', `Done for ...${shortId}!`);
        processedCount++;
        
        console.log('\n' + '═'.repeat(60));
        console.log(`👤 User: ${senderName} (...${shortId})`);
        if (isFollowUp) console.log(`🔄 FOLLOW-UP RESPONSE`);
        console.log(`📊 ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.video} video, ${counts.texts} text`);
        console.log(`⏰ ${new Date().toLocaleString()}`);
        console.log('═'.repeat(60));
        console.log(primaryResponseText);
        console.log('═'.repeat(60) + '\n');
        
        await sock.sendPresenceUpdate('composing', destinationChatId);
        const delay = Math.floor(Math.random() * (CONFIG.TYPING_DELAY_MAX - CONFIG.TYPING_DELAY_MIN)) + CONFIG.TYPING_DELAY_MIN;
        await new Promise(resolve => setTimeout(resolve, delay));
        await sock.sendPresenceUpdate('paused', destinationChatId);
        
        let finalResponseText = primaryResponseText.length <= 3800 
            ? primaryResponseText 
            : primaryResponseText.substring(0, 3800) + '\n\n_(truncated)_';
        
        // 🟢 FEATURE ADDITION: Put captions at the top if auto-forwarding to a target group
        if (targetChatId && allOriginalText.length > 0) {
            const captionHeader = allOriginalText.map(t => t.replace(/^\[.*?\]:\s*/, '')).join('\n');
            if (captionHeader.trim().length > 0) {
                finalResponseText = `${captionHeader}\n\n${finalResponseText}`;
            }
        }

        const sentMessage = await sock.sendMessage(destinationChatId, { 
            text: finalResponseText,
            mentions: [senderId]
        });
        
        if (sentMessage?.key?.id) {
            const messageId = sentMessage.key.id;
            trackBotMessage(destinationChatId, messageId);
            storeContext(destinationChatId, messageId, mediaFiles, primaryResponseText, senderId);
            log('💾', `Context stored for ...${shortId}`);
        }
        
        log('📤', `Sent to target/chat!`);
        
    } catch (error) {
        log('❌', `Error for ...${shortId}: ${error.message}`);
        console.error(error);
        
        // 🟢 MODIFICATION START: 5-Minute Auto-Retry Logic
        if (retryAttempt === 0) {
            log('⏳', `Generation failed. Scheduling retry in 5 mins for ...${shortId}`);
            
            await sock.sendPresenceUpdate('composing', destinationChatId);
            await new Promise(r => setTimeout(r, 1000));
            
            await sock.sendMessage(destinationChatId, { 
                text: `⚠️ *High Traffic / Network Alert*\n\nThe AI model is currently overloaded/unstable. I have queued your request and will *automatically retry in 5 minutes*.\n\nPlease do not resend the files.`,
                mentions: [senderId]
            });
            
            // Schedule the retry with exact same parameters, but increment retryAttempt
            setTimeout(() => {
                log('🔄', `Executing 5-minute retry for ...${shortId}`);
                processMedia(sock, chatId, mediaFiles, isFollowUp, previousResponse, senderId, senderName, userTextInput, targetFps, isSecondaryMode, targetChatId, 1);
            }, 300000); // 300,000 ms = 5 minutes
            
            return;
        }
        // 🟢 MODIFICATION END

        await sock.sendPresenceUpdate('composing', destinationChatId);
        await new Promise(r => setTimeout(r, 1500));
        
        await sock.sendMessage(destinationChatId, { 
            text: `❌ @${senderId.split('@')[0]}, error processing your request:\n_${error.message}_\n\nPlease try again later.`,
            mentions: [senderId]
        });
    }
}

console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║         WhatsApp Clinical Profile Bot v3.0                 ║');
console.log('║                                                            ║');
console.log('║  📷 Images  📄 PDFs  🎤 Voice  🎵 Audio  🎬 Video  💬 Text ║');
console.log('║                                                            ║');
console.log('║  🌍 UNIVERSAL MODE: Works in any chat (Group or Private)  ║');
console.log('║  🔄 AUTO-GROUPS: Monitors Source -> Sends to Target (60s) ║');
console.log('║  🔀 SMART BATCHING: Splits distinct patients automatically║');
console.log('║  🎥 SMART VIDEO: Oversamples & Picks Sharpest Frames      ║');
console.log('║     Use: . (3fps), .2 (2fps), .1 (1fps)                   ║');
console.log('║  🧠 SECONDARY ANALYSIS: Use .. (double dot) for Chain     ║');
console.log('║                                                            ║');
console.log('║  ✨ Per-User Buffers - Each user processed separately     ║');
console.log('║  ↩️ Reply to ask questions OR add context                  ║');
console.log('║  🗄️ MongoDB Persistent Sessions                           ║');
console.log('║  🔑 Multi-Key Rotation (2hrs) + Failover Active           ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

log('🏁', 'Starting...');

if (CONFIG.API_KEYS.length === 0) {
    log('❌', 'No API Keys found! Set GEMINI_API_KEYS environment variable.');
} else {
    log('🔑', `Loaded ${CONFIG.API_KEYS.length} Gemini API Key(s)`);
}

(async () => {
    try {
        await connectMongoDB();
        await startBot();
    } catch (error) {
        log('💥', `Startup error: ${error.message}`);
        console.error(error);
        process.exit(1);
    }
})();
