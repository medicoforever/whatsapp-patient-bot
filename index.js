import { GoogleGenerativeAI } from '@google/generative-ai';
import pino from 'pino';
import QRCode from 'qrcode';
import express from 'express';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import mongoose from 'mongoose';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CONFIG = {
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    GEMINI_MODEL: 'gemini-2.0-flash',
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
    SYSTEM_INSTRUCTION: `You are an expert medical AI assistant specializing in radiology and clinical documentation. You have two modes of operation:

**MODE 1 - CLINICAL PROFILE GENERATION:**
When asked to generate or update a Clinical Profile, extract transcript/raw text from uploaded files (images, PDFs, audio recordings, or video files) and any text context provided. Create a concise and comprehensive "Clinical Profile".

IMPORTANT: IF HANDWRITTEN TEXT IS NOT LEGIBLE, USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER IT.

FOR AUDIO FILES: Transcribe the audio content carefully and extract all relevant medical information.
FOR VIDEO FILES: Analyze the video content, transcribe any audio, and extract all visible medical information.
FOR TEXT MESSAGES: Incorporate additional clinical context, patient history, or notes into the Clinical Profile.

Follow these instructions for Clinical Profile generation:
- Analyze all provided content meticulously
- Extract key information: scan types, dates, findings, measurements, impressions, clinical history
- Synthesize into a single cohesive paragraph
- Frame sentences concisely but DO NOT omit important clinical details
- EXCLUDE patient's name, age, or gender
- Arrange dated scan reports chronologically in ascending order
- For scans without dates, refer to them as "Previous [Scan Type]..."
- Output format: A single paragraph starting with "*Clinical Profile:" wrapped in asterisks

**MODE 2 - FOLLOW-UP QUESTIONS AND CLARIFICATIONS:**
When a user asks a question or requests clarification about a previously generated Clinical Profile:
- Answer the question directly and informatively
- Provide medical explanations in clear, understandable language
- Reference the Clinical Profile and attached files as needed
- Do NOT regenerate the Clinical Profile unless explicitly asked
- Be helpful, professional, and thorough in your response

Determine which mode to use based on the user's request.`
};

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

// Helper function to detect if text is a question or request
function isQuestionOrRequest(text) {
    if (!text) return false;
    const lowerText = text.toLowerCase().trim();
    
    // Check for question marks
    if (lowerText.includes('?')) return true;
    
    // Check for question words at the start
    const questionStarters = [
        'what', 'why', 'how', 'when', 'where', 'who', 'which',
        'is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
        'can ', 'could ', 'would ', 'should ', 'will ', 'shall ',
        'have ', 'has ', 'had ',
        'explain', 'tell me', 'describe', 'clarify', 'elaborate',
        'meaning', 'means', 'mean ',
        'significance', 'significant',
        'serious', 'concern', 'worried', 'normal', 'abnormal',
        'treatment', 'therapy', 'medication', 'medicine',
        'diagnosis', 'prognosis', 'cause', 'reason',
        'next step', 'recommend', 'suggestion', 'advice',
        'please explain', 'please tell', 'please clarify',
        'i want to know', 'i need to know', 'i\'d like to know',
        'can you', 'could you', 'would you', 'please'
    ];
    
    for (const starter of questionStarters) {
        if (lowerText.startsWith(starter) || lowerText.includes(' ' + starter)) {
            return true;
        }
    }
    
    return false;
}

// Helper function to extract text from any message type
function extractTextFromMessage(msg) {
    const messageType = Object.keys(msg.message || {})[0];
    
    if (!messageType) return '';
    
    switch (messageType) {
        case 'conversation':
            return msg.message.conversation || '';
        case 'extendedTextMessage':
            return msg.message.extendedTextMessage?.text || '';
        case 'imageMessage':
            return msg.message.imageMessage?.caption || '';
        case 'videoMessage':
            return msg.message.videoMessage?.caption || '';
        case 'documentMessage':
            return msg.message.documentMessage?.caption || '';
        default:
            return '';
    }
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
            .warning { background: #FF9800; }
            .error { background: #f44336; }
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
            .info-box code {
                background: rgba(0,0,0,0.3);
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 11px;
                word-break: break-all;
            }
            .group-list {
                text-align: left;
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 12px;
                margin-top: 15px;
            }
            .group-list h4 { margin: 0 0 10px 0; }
            .group-item {
                background: rgba(0,0,0,0.2);
                padding: 10px;
                border-radius: 8px;
                margin: 8px 0;
                font-size: 12px;
            }
            .group-item strong { display: block; margin-bottom: 5px; }
            .group-item code {
                background: rgba(0,0,0,0.3);
                padding: 3px 6px;
                border-radius: 4px;
                font-size: 10px;
                word-break: break-all;
                display: block;
            }
            .copy-btn {
                background: rgba(255,255,255,0.2);
                border: none;
                color: white;
                padding: 3px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 10px;
                margin-top: 5px;
            }
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
            .configured {
                background: rgba(76, 175, 80, 0.3);
                border: 1px solid rgba(76, 175, 80, 0.5);
                padding: 10px;
                border-radius: 8px;
                margin: 15px 0;
            }
            .not-configured {
                background: rgba(255, 152, 0, 0.3);
                border: 1px solid rgba(255, 152, 0, 0.5);
                padding: 10px;
                border-radius: 8px;
                margin: 15px 0;
            }
            .db-status {
                font-size: 11px;
                padding: 5px 10px;
                border-radius: 20px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .db-connected { background: rgba(76, 175, 80, 0.3); }
            .db-disconnected { background: rgba(244, 67, 54, 0.3); }
            .media-support {
                background: rgba(0,0,0,0.15);
                padding: 10px;
                border-radius: 8px;
                margin-top: 10px;
                font-size: 12px;
            }
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
            <p class="subtitle">Medical Clinical Profile Generator (Per-User Buffers)</p>
            <div class="db-status ${mongoConnected ? 'db-connected' : 'db-disconnected'}">
                ${mongoConnected ? '🗄️ MongoDB Connected (Persistent Sessions)' : '⚠️ MongoDB Not Connected'}
            </div>
    `;
    
    if (isConnected) {
        if (CONFIG.ALLOWED_GROUP_ID) {
            const groupName = discoveredGroups.get(CONFIG.ALLOWED_GROUP_ID) || 'Configured Group';
            html += `
                <div class="status connected">✅ ACTIVE IN GROUP</div>
                <div class="configured">
                    <strong>🎯 Active Group:</strong> ${groupName}<br>
                    <code style="font-size:10px">${CONFIG.ALLOWED_GROUP_ID}</code>
                </div>
                <div class="stats">
                    <div class="stat"><div class="stat-value">${stats.users}</div><div class="stat-label">👥 Users</div></div>
                    <div class="stat"><div class="stat-value">${stats.images}</div><div class="stat-label">📷 Images</div></div>
                    <div class="stat"><div class="stat-value">${stats.pdfs}</div><div class="stat-label">📄 PDFs</div></div>
                    <div class="stat"><div class="stat-value">${stats.audio}</div><div class="stat-label">🎤 Audio</div></div>
                    <div class="stat"><div class="stat-value">${stats.video}</div><div class="stat-label">🎬 Video</div></div>
                    <div class="stat"><div class="stat-value">${stats.texts}</div><div class="stat-label">💬 Texts</div></div>
                    <div class="stat"><div class="stat-value">${storedContextsCount}</div><div class="stat-label">🧠 Ctx</div></div>
                    <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">✅ Done</div></div>
                </div>
                <div class="info-box">
                    <h3>✨ Usage:</h3>
                    <p><strong>New Request:</strong><br>
                    1. Send files/text → Send <strong>.</strong> → Get profile<br><br>
                    <strong>👥 Multi-User Support:</strong><br>
                    Each user's files are processed separately!<br><br>
                    <strong>↩️ Reply Feature:</strong><br>
                    Reply to bot's response to ask questions or add context!</p>
                </div>
                <div class="media-support">
                    <span class="feature-badge">📷 Images</span>
                    <span class="feature-badge">📄 PDFs</span>
                    <span class="feature-badge">🎤 Voice</span>
                    <span class="feature-badge">🎵 MP3/WAV</span>
                    <span class="feature-badge">🎬 MP4/Video</span>
                    <span class="feature-badge">💬 Text</span>
                    <span class="feature-badge">❓ Q&A</span>
                </div>
            `;
        } else {
            html += `
                <div class="status warning">⚠️ DISCOVERY MODE</div>
                <div class="not-configured">
                    <strong>No group configured yet!</strong><br>
                    Send a message in your target group to discover its ID.
                </div>
            `;
            
            if (discoveredGroups.size > 0) {
                html += `<div class="group-list"><h4>📋 Discovered Groups:</h4>`;
                for (const [id, name] of discoveredGroups) {
                    html += `
                        <div class="group-item">
                            <strong>${name}</strong>
                            <code>${id}</code>
                            <button class="copy-btn" onclick="navigator.clipboard.writeText('${id}')">📋 Copy ID</button>
                        </div>
                    `;
                }
                html += `</div>`;
                
                html += `
                    <div class="info-box">
                        <h3>🔧 Next Steps:</h3>
                        <p>1. Copy the Group ID above<br>
                        2. Go to Render Dashboard → Environment<br>
                        3. Add: <code>ALLOWED_GROUP_ID</code> = (paste ID)<br>
                        4. Click "Save Changes" and redeploy</p>
                    </div>
                `;
            }
        }
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
    const stats = getTotalBufferStats(CONFIG.ALLOWED_GROUP_ID);
    
    const storedContextsCount = chatContexts.has(CONFIG.ALLOWED_GROUP_ID) 
        ? chatContexts.get(CONFIG.ALLOWED_GROUP_ID).size 
        : 0;
    
    res.json({ 
        status: 'running',
        connected: isConnected,
        mongoConnected: mongoConnected,
        configuredGroup: CONFIG.ALLOWED_GROUP_ID || 'NOT SET',
        discoveredGroups: Object.fromEntries(discoveredGroups),
        bufferedMedia: stats,
        storedContexts: storedContextsCount,
        processedCount,
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
                
                if (CONFIG.ALLOWED_GROUP_ID) {
                    log('🎯', `Bot active ONLY in: ${CONFIG.ALLOWED_GROUP_ID}`);
                } else {
                    log('⚠️', 'No group configured! Send a message in target group to get its ID.');
                }
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
    
    const isGroup = chatId.endsWith('@g.us');
    const senderId = getSenderId(msg);
    const senderName = getSenderName(msg);
    const shortId = getShortSenderId(senderId);
    
    if (isGroup && !discoveredGroups.has(chatId)) {
        try {
            const metadata = await sock.groupMetadata(chatId);
            discoveredGroups.set(chatId, metadata.subject);
            log('📋', `Discovered group: "${metadata.subject}"`);
            log('📋', `Group ID: ${chatId}`);
            console.log('\n' + '='.repeat(50));
            console.log('🎯 TO USE THIS GROUP, ADD THIS ENVIRONMENT VARIABLE:');
            console.log(`   ALLOWED_GROUP_ID = ${chatId}`);
            console.log('='.repeat(50) + '\n');
        } catch (e) {
            discoveredGroups.set(chatId, 'Unknown Group');
        }
    }
    
    if (!isAllowedGroup(chatId)) {
        return;
    }
    
    const messageType = Object.keys(msg.message)[0];
    
    let quotedMessageId = null;
    let contextInfo = null;
    
    // Extract context info from various message types
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
    } else if (messageType === 'conversation') {
        // Conversation type doesn't typically have contextInfo for replies
        // but check anyway
        contextInfo = msg.message.conversation?.contextInfo;
    }
    
    if (contextInfo?.stanzaId) {
        quotedMessageId = contextInfo.stanzaId;
        
        if (isBotMessage(chatId, quotedMessageId)) {
            log('↩️', `Reply to bot from ${senderName} (...${shortId})`);
            await handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName);
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
        
        if (text === CONFIG.TRIGGER_TEXT) {
            log('🔔', `Trigger from ${senderName} (...${shortId})`);
            
            await new Promise(r => setTimeout(r, 1000));
            
            const userBufferCount = getUserBufferCount(chatId, senderId);
            
            if (userBufferCount > 0) {
                clearUserTimeout(chatId, senderId);
                const mediaFiles = clearUserBuffer(chatId, senderId);
                
                log('🤖', `Processing ${mediaFiles.length} item(s) for ...${shortId}`);
                await processMedia(sock, chatId, mediaFiles, false, null, senderId, senderName);
            } else {
                await sock.sendMessage(chatId, { 
                    text: `ℹ️ @${senderId.split('@')[0]}, you have no files buffered.\n\nSend images, PDFs, audio (.mp3, .wav), video (.mp4), or text first, then send *.*\n\n💡 _Or reply to my previous response to ask questions or add context!_`,
                    mentions: [senderId]
                });
            }
        }
        else if (text.toLowerCase() === 'help' || text === '?') {
            await sock.sendMessage(chatId, { 
                text: `🏥 *Clinical Profile Bot*\n\n*Supported Files:*\n📷 Images (photos, scans)\n📄 PDFs (reports, documents)\n🎤 Voice messages\n🎵 Audio files (.mp3, .wav, .ogg, .m4a)\n🎬 Video files (.mp4, .mkv, .avi, .mov)\n💬 Text notes & captions\n\n*Basic Usage:*\n1️⃣ Send file(s) and/or text\n2️⃣ Send *.* to process\n\n*👥 Multi-User:*\nEach user's files are tracked separately!\n\n*↩️ Reply Feature:*\n• Reply to my response to ask questions\n• Reply with more files/text to refine the profile\n\n*Commands:*\n• *.* - Process YOUR content\n• *clear* - Clear YOUR buffer\n• *status* - Check status` 
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
                text: `📊 *Status*\n\n*Your Buffer:* ${userCount} item(s)\n\n*Group Total:*\n👥 Active users: ${stats.users}\n📷 Images: ${stats.images}\n📄 PDFs: ${stats.pdfs}\n🎵 Audio: ${stats.audio}\n🎬 Video: ${stats.video}\n💬 Texts: ${stats.texts}\n━━━━━━━━━━\n📦 Total buffered: ${stats.total}\n🧠 Stored contexts: ${storedContexts}\n✅ Processed: ${processedCount}\n🗄️ MongoDB: ${mongoConnected ? 'Connected' : 'Not connected'}` 
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

async function handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName) {
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
    
    const messageType = Object.keys(msg.message)[0];
    const newContent = [];
    let replyText = '';
    
    // Handle conversation type (simple text reply)
    if (messageType === 'conversation') {
        const text = msg.message.conversation || '';
        replyText = text.trim();
        if (replyText) {
            newContent.push({
                type: 'text',
                content: replyText,
                sender: senderName,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('💬', `Follow-up text (conversation) from ...${shortId}: "${replyText.substring(0, 50)}${replyText.length > 50 ? '...' : ''}"`);
        }
    }
    // Handle extended text message (text with reply context)
    else if (messageType === 'extendedTextMessage') {
        const text = msg.message.extendedTextMessage?.text || '';
        replyText = text.trim();
        if (replyText) {
            newContent.push({
                type: 'text',
                content: replyText,
                sender: senderName,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('💬', `Follow-up text from ...${shortId}: "${replyText.substring(0, 50)}${replyText.length > 50 ? '...' : ''}"`);
        }
    }
    // Handle image reply
    else if (messageType === 'imageMessage') {
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const caption = msg.message.imageMessage.caption || '';
            replyText = caption;
            
            newContent.push({
                type: 'image',
                data: buffer.toString('base64'),
                mimeType: msg.message.imageMessage.mimetype || 'image/jpeg',
                caption: caption,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('📷', `Follow-up image from ...${shortId}`);
            
            if (caption) {
                newContent.push({
                    type: 'text',
                    content: caption,
                    sender: senderName,
                    timestamp: Date.now(),
                    isFollowUp: true,
                    isCaption: true
                });
            }
        } catch (error) {
            log('❌', `Image error: ${error.message}`);
        }
    }
    // Handle video reply
    else if (messageType === 'videoMessage') {
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            const caption = msg.message.videoMessage.caption || '';
            replyText = caption;
            
            newContent.push({
                type: 'video',
                data: buffer.toString('base64'),
                mimeType: msg.message.videoMessage.mimetype || 'video/mp4',
                caption: caption,
                duration: msg.message.videoMessage.seconds || 0,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('🎬', `Follow-up video from ...${shortId}`);
            
            if (caption) {
                newContent.push({
                    type: 'text',
                    content: caption,
                    sender: senderName,
                    timestamp: Date.now(),
                    isFollowUp: true,
                    isCaption: true
                });
            }
        } catch (error) {
            log('❌', `Video error: ${error.message}`);
        }
    }
    // Handle audio reply
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
    // Handle document reply
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        const caption = msg.message.documentMessage.caption || '';
        replyText = caption;
        
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
    const isQuestion = isQuestionOrRequest(replyText);
    
    const combinedMedia = [...storedContext.mediaFiles, ...newContent];
    
    log('🔄', `Processing reply for ...${shortId}: ${storedContext.mediaFiles.length} original + ${newContent.length} new (isQuestion: ${isQuestion})`);
    
    await processMedia(sock, chatId, combinedMedia, true, storedContext.response, senderId, senderName, isQuestion, replyText);
}

let model;
try {
    const genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);
    model = genAI.getGenerativeModel({ 
        model: CONFIG.GEMINI_MODEL,
        systemInstruction: CONFIG.SYSTEM_INSTRUCTION
    });
    log('✅', 'Gemini AI ready');
} catch (error) {
    log('❌', `Gemini init error: ${error.message}`);
}

async function processMedia(sock, chatId, mediaFiles, isFollowUp = false, previousResponse = null, senderId, senderName, isQuestion = false, userQuery = '') {
    const shortId = getShortSenderId(senderId);
    
    try {
        const counts = { images: 0, pdfs: 0, audio: 0, video: 0, texts: 0, followUps: 0 };
        const textContents = [];
        const captions = [];
        const binaryMedia = [];
        const followUpTexts = [];
        
        mediaFiles.forEach(m => {
            if (m.isFollowUp) counts.followUps++;
            
            if (m.type === 'image') {
                counts.images++;
                binaryMedia.push(m);
                if (m.caption && !m.isFollowUp) {
                    captions.push(`[Image caption]: ${m.caption}`);
                }
            }
            else if (m.type === 'pdf') {
                counts.pdfs++;
                binaryMedia.push(m);
                if (m.caption && !m.isFollowUp) {
                    captions.push(`[PDF caption]: ${m.caption}`);
                }
            }
            else if (m.type === 'audio' || m.type === 'voice') {
                counts.audio++;
                binaryMedia.push(m);
                if (m.caption && !m.isFollowUp) {
                    captions.push(`[Audio caption]: ${m.caption}`);
                }
            }
            else if (m.type === 'video') {
                counts.video++;
                binaryMedia.push(m);
                if (m.caption && !m.isFollowUp) {
                    captions.push(`[Video caption]: ${m.caption}`);
                }
            }
            else if (m.type === 'text') {
                counts.texts++;
                if (m.isFollowUp && !m.isCaption) {
                    followUpTexts.push(`[User's follow-up message]: ${m.content}`);
                } else if (!m.isFollowUp) {
                    textContents.push(`[Text note from ${m.sender}]: ${m.content}`);
                }
            }
        });
        
        if (isFollowUp) {
            log('🤖', `Processing follow-up for ...${shortId}: ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.video} video, ${counts.texts} text (question: ${isQuestion})`);
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
            // Check if user is asking a question or providing additional context
            if (isQuestion && userQuery) {
                // User is asking a question about the Clinical Profile
                promptText = `The user is asking a QUESTION or making a REQUEST about the previously generated Clinical Profile. Please answer their question directly and helpfully.

=== PREVIOUS CLINICAL PROFILE YOU GENERATED ===
${previousResponse}
=== END PREVIOUS CLINICAL PROFILE ===

=== USER'S QUESTION/REQUEST ===
${userQuery}
=== END QUESTION ===

${followUpTexts.length > 0 ? `=== ADDITIONAL CONTEXT FROM USER ===\n${followUpTexts.join('\n\n')}\n=== END ADDITIONAL CONTEXT ===\n\n` : ''}
Please provide a helpful, informative response to the user's question. You may reference the Clinical Profile and any attached medical files. 

IMPORTANT: 
- Answer the question directly and conversationally
- Provide clear medical explanations in understandable language
- If the user asks about significance, prognosis, or recommendations, provide appropriate medical guidance
- Do NOT regenerate the Clinical Profile unless explicitly asked to do so
- Be professional, thorough, and helpful`;
            } else {
                // User is providing additional context to refine the profile
                promptText = `This is a FOLLOW-UP request to refine the Clinical Profile. The user has provided additional context or information.

=== PREVIOUS CLINICAL PROFILE GENERATED ===
${previousResponse}
=== END PREVIOUS RESPONSE ===

=== ORIGINAL CONTEXT ===
${allOriginalText.length > 0 ? allOriginalText.join('\n\n') : '(Original files are attached)'}
=== END ORIGINAL CONTEXT ===

=== NEW ADDITIONAL INFORMATION FROM USER ===
${followUpTexts.join('\n\n')}
=== END NEW INFORMATION ===

Please analyze ALL the content (original files + original text + NEW additional information) and generate an UPDATED Clinical Profile that incorporates the new information. 

The Clinical Profile should be a single paragraph starting with "*Clinical Profile:" and wrapped in single asterisks. Maintain the same format as before but include the new information appropriately.`;
            }
        } else if (binaryMedia.length > 0 && allOriginalText.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} along with the following additional text notes/context, and generate the Clinical Profile.

=== ADDITIONAL TEXT NOTES ===
${allOriginalText.join('\n\n')}
=== END OF TEXT NOTES ===

For audio files, transcribe the content first, then extract medical information.
For video files, analyze visual content and transcribe any audio.`;
        } 
        else if (binaryMedia.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} containing medical information and generate the Clinical Profile. For audio files, transcribe the content first. For video files, analyze visual content and transcribe any audio.`;
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
        
        // Add safety check for empty prompt
        if (!promptText || promptText.trim() === '') {
            log('⚠️', `Empty prompt for ...${shortId}`);
            await sock.sendMessage(chatId, { 
                text: `⚠️ @${senderId.split('@')[0]}, I couldn't process your request. Please try again with more content.`,
                mentions: [senderId]
            });
            return;
        }
        
        log('📝', `Sending to Gemini (prompt length: ${promptText.length}, media items: ${contentParts.length})`);
        
        const result = await model.generateContent(requestContent);
        let responseText = result.response.text();
        
        // Check for empty response
        if (!responseText || responseText.trim() === '') {
            log('⚠️', `Empty response from Gemini for ...${shortId}`);
            responseText = `I apologize, but I couldn't generate a proper response. Please try again or provide more context.`;
        }
        
        log('✅', `Response received for ...${shortId} (length: ${responseText.length})`);
        processedCount++;
        
        console.log('\n' + '═'.repeat(60));
        console.log(`👤 User: ${senderName} (...${shortId})`);
        if (isFollowUp) {
            console.log(`🔄 FOLLOW-UP ${isQuestion ? '(QUESTION)' : '(ADDITIONAL CONTEXT)'}`);
        }
        console.log(`📊 ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.video} video, ${counts.texts} text`);
        console.log(`⏰ ${new Date().toLocaleString()}`);
        console.log('═'.repeat(60));
        console.log(responseText);
        console.log('═'.repeat(60) + '\n');
        
        await sock.sendPresenceUpdate('composing', chatId);
        const delay = Math.floor(Math.random() * (CONFIG.TYPING_DELAY_MAX - CONFIG.TYPING_DELAY_MIN)) + CONFIG.TYPING_DELAY_MIN;
        await new Promise(resolve => setTimeout(resolve, delay));
        await sock.sendPresenceUpdate('paused', chatId);
        
        let sentMessage;
        const finalResponse = responseText.length <= 3800 
            ? responseText 
            : responseText.substring(0, 3800) + '\n\n_(truncated)_';
        
        sentMessage = await sock.sendMessage(chatId, { 
            text: finalResponse,
            mentions: [senderId]
        });
        
        if (sentMessage?.key?.id) {
            const messageId = sentMessage.key.id;
            trackBotMessage(chatId, messageId);
            storeContext(chatId, messageId, mediaFiles, responseText, senderId);
            log('💾', `Context stored for ...${shortId}`);
        }
        
        log('📤', `Sent to ...${shortId}!`);
        
    } catch (error) {
        log('❌', `Error for ...${shortId}: ${error.message}`);
        console.error(error);
        
        await sock.sendPresenceUpdate('composing', chatId);
        await new Promise(r => setTimeout(r, 1500));
        
        let errorMessage = error.message;
        if (error.message.includes('SAFETY')) {
            errorMessage = 'Content was flagged by safety filters. Please try with different content.';
        } else if (error.message.includes('quota') || error.message.includes('limit')) {
            errorMessage = 'API quota exceeded. Please try again later.';
        } else if (error.message.includes('timeout')) {
            errorMessage = 'Request timed out. Please try again.';
        }
        
        await sock.sendMessage(chatId, { 
            text: `❌ @${senderId.split('@')[0]}, error processing your request:\n_${errorMessage}_\n\nPlease try again.`,
            mentions: [senderId]
        });
    }
}

console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║         WhatsApp Clinical Profile Bot v2.2                 ║');
console.log('║                                                            ║');
console.log('║  📷 Images  📄 PDFs  🎤 Voice  🎵 Audio  🎬 Video  💬 Text ║');
console.log('║                                                            ║');
console.log('║  🎵 Supports: MP3, WAV, OGG, M4A, AAC, FLAC               ║');
console.log('║  🎬 Supports: MP4, MKV, AVI, MOV, WEBM                    ║');
console.log('║                                                            ║');
console.log('║  ✨ Per-User Buffers - Each user processed separately     ║');
console.log('║  ↩️ Reply to bot response to ask questions or add context  ║');
console.log('║  ❓ Smart Q&A - Detects questions vs additional context   ║');
console.log('║  🗄️ MongoDB Persistent Sessions                           ║');
console.log('║                                                            ║');
console.log('║  🔒 Works ONLY in ONE specific group                       ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

log('🏁', 'Starting...');

if (!CONFIG.GEMINI_API_KEY) {
    log('❌', 'GEMINI_API_KEY not set!');
}

if (CONFIG.ALLOWED_GROUP_ID) {
    log('🎯', `Configured for group: ${CONFIG.ALLOWED_GROUP_ID}`);
} else {
    log('⚠️', 'No group configured! Bot will discover groups when you message them.');
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
