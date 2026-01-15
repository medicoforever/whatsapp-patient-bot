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

// ============== CONFIGURATION ==============
const CONFIG = {
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    GEMINI_MODEL: 'gemini-3-flash-preview',
    MONGODB_URI: process.env.MONGODB_URI,
    ALLOWED_GROUP_ID: process.env.ALLOWED_GROUP_ID || '',
    MEDIA_TIMEOUT_MS: 300000, // 5 minutes
    CONTEXT_RETENTION_MS: 1800000, // 30 minutes
    MAX_STORED_CONTEXTS: 20,
    TRIGGER_TEXT: '.',
    COMMANDS: ['.', 'help', '?', 'clear', 'status'],
    TYPING_DELAY_MIN: 3000,
    TYPING_DELAY_MAX: 6000,
    SYSTEM_INSTRUCTION: `You are an expert medical AI assistant specializing in radiology. You have to extract transcript / raw text from one or more uploaded files (images, PDFs, or audio recordings). You may also receive additional text context provided by the user. Your task is to analyze all content to create a concise and comprehensive "Clinical Profile".

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT

FOR AUDIO FILES: Transcribe the audio content carefully and extract all relevant medical information mentioned.

FOR TEXT MESSAGES: These may contain additional clinical context, patient history, or notes that should be incorporated into the Clinical Profile.

FOR FOLLOW-UP REQUESTS: If the user provides additional context or corrections after receiving a Clinical Profile, incorporate that new information and regenerate an updated Clinical Profile.

YOUR RESPONSE MUST BE BASED SOLELY ON THE PROVIDED CONTENT (files AND text).

Follow these strict instructions:

Analyze All Content: Meticulously examine all provided files - images, PDFs, and audio recordings, as well as any accompanying text messages. This may include prior medical scan reports (like USG, CT, MRI), clinical notes, voice memos, or other relevant documents.

Extract Key Information: From the content, identify and extract all pertinent information, such as:

Scan types (e.g., USG, CT Brain).

Dates of scans or documents.

Key findings, measurements, or impressions from reports.

Relevant clinical history mentioned in notes, audio, or text messages.

Synthesize into a Clinical Profile:

Combine all extracted information into a single, cohesive paragraph. This represents a 100% recreation of the relevant clinical details from the provided content.

If there are repeated or vague findings across multiple documents, synthesize them into a single, concise statement.

Frame sentences properly to be concise, but you MUST NOT omit any important clinical details. Prioritize completeness of clinical information over extreme brevity.

You MUST strictly exclude any mention of the patient's name, age, or gender.

If multiple dated scan reports are present, you MUST arrange their summaries chronologically in ascending order based on their dates.

If a date is not available for a scan, refer to it as "Previous [Scan Type]...".

Formatting:

The final output MUST be a single paragraph.

This paragraph MUST start with "Clinical Profile:" and the entire content (including the prefix) must be wrapped in single asterisks. For example: "*Clinical Profile: Previous USG dated 01/01/2023 showed mild hepatomegaly. Patient also has a H/o hypertension as noted in the clinical sheet.*"

Output:

Do not output the raw transcribed text.

Do not output JSON or Markdown code blocks.

Return ONLY the single formatted paragraph described above.

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT`
};

// ============== MONGODB SETUP ==============
const sessionSchema = new mongoose.Schema({
    key: { type: String, required: true, unique: true },
    value: { type: mongoose.Schema.Types.Mixed, required: true },
    updatedAt: { type: Date, default: Date.now }
}, { collection: 'whatsapp_sessions' });

sessionSchema.index({ updatedAt: 1 }, { expireAfterSeconds: 86400 * 30 }); // 30 days TTL

let SessionModel;

// ============== MONGODB AUTH STATE ==============
async function useMongoDBAuthState() {
    // Ensure model is created
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

    // Load existing creds or create new
    let creds = await readData('auth_creds');
    
    if (!creds) {
        // Import and initialize fresh credentials
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

// ============== SETUP ==============
// Per-User Media Buffers: chatId -> senderId -> [media files]
const chatMediaBuffers = new Map();
// Per-User Timeouts: chatId -> senderId -> timeout
const chatTimeouts = new Map();
// Discovered groups
const discoveredGroups = new Map();
// Context storage for reply feature: chatId -> Map(messageId -> {mediaFiles, response, timestamp, senderId})
const chatContexts = new Map();
// Track bot's own message IDs: chatId -> Set of message IDs
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

// Get sender ID from message
function getSenderId(msg) {
    return msg.key.participant || msg.key.remoteJid;
}

// Get sender name (phone number in readable format)
function getSenderName(msg) {
    const senderId = getSenderId(msg);
    // Extract phone number from JID
    const phone = senderId.split('@')[0];
    return msg.pushName || phone;
}

// Get short sender ID for logging
function getShortSenderId(senderId) {
    const phone = senderId.split('@')[0];
    if (phone.length > 6) {
        return phone.slice(-4); // Last 4 digits
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

// ============== PER-USER BUFFER MANAGEMENT ==============
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
    const stats = { users: 0, images: 0, pdfs: 0, audio: 0, texts: 0, total: 0 };
    if (!chatMediaBuffers.has(chatId)) return stats;
    
    const chatBuffer = chatMediaBuffers.get(chatId);
    stats.users = chatBuffer.size;
    
    for (const [senderId, items] of chatBuffer) {
        items.forEach(m => {
            if (m.type === 'image') stats.images++;
            else if (m.type === 'pdf') stats.pdfs++;
            else if (m.type === 'audio' || m.type === 'voice') stats.audio++;
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
    
    // Clear existing timeout for this user
    if (chatTimeoutMap.has(senderId)) {
        clearTimeout(chatTimeoutMap.get(senderId));
    }
    
    // Set new timeout
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

// ============== CONTEXT MANAGEMENT ==============
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

// ============== WEB SERVER ==============
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
                max-width: 550px;
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
                gap: 8px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            .stat {
                background: rgba(255,255,255,0.1);
                padding: 10px 12px;
                border-radius: 10px;
                min-width: 55px;
            }
            .stat-value { font-size: 18px; font-weight: bold; }
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
                    <div class="stat"><div class="stat-value">${stats.texts}</div><div class="stat-label">💬 Texts</div></div>
                    <div class="stat"><div class="stat-value">${storedContextsCount}</div><div class="stat-label">🧠 Contexts</div></div>
                    <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">✅ Done</div></div>
                </div>
                <div class="info-box">
                    <h3>✨ Usage:</h3>
                    <p><strong>New Request:</strong><br>
                    1. Send files/text → Send <strong>.</strong> → Get profile<br><br>
                    <strong>👥 Multi-User Support:</strong><br>
                    Each user's files are processed separately!<br><br>
                    <strong>↩️ Reply Feature:</strong><br>
                    Reply to bot's response to add more context!</p>
                </div>
                <div class="media-support">
                    <span class="feature-badge">📷 Images</span>
                    <span class="feature-badge">📄 PDFs</span>
                    <span class="feature-badge">🎤 Voice</span>
                    <span class="feature-badge">💬 Text</span>
                    <span class="feature-badge">👥 Per-User</span>
                    <span class="feature-badge">🗄️ Persistent</span>
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

// ============== LOAD BAILEYS ==============
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

// ============== MONGODB CONNECTION ==============
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

// ============== WHATSAPP BOT ==============
async function startBot() {
    try {
        botStatus = 'Initializing...';
        log('🚀', 'Starting WhatsApp Bot...');
        
        if (!makeWASocket) await loadBaileys();
        
        // Get auth state
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
            // Fallback to file auth (will lose session on restart)
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
        
        // Store auth state for later
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

        // Connection updates
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
                
                // Check if logged out
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
                    // Reconnect for other errors
                    const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
                    log('🔄', `Reconnecting in 5 seconds... (shouldReconnect: ${shouldReconnect})`);
                    setTimeout(startBot, 5000);
                }
                
            } else if (connection === 'open') {
                isConnected = true;
                qrCodeDataURL = null;
                botStatus = 'Connected';
                
                log('✅', '🎉 CONNECTED TO WHATSAPP!');
                
                // Save credentials immediately
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

        // Credentials update
        sock.ev.on('creds.update', async () => {
            if (authState?.saveCreds) {
                await authState.saveCreds();
            }
        });

        // Messages
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

// ============== MESSAGE HANDLER ==============
async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    
    if (chatId === 'status@broadcast') return;
    
    const isGroup = chatId.endsWith('@g.us');
    const senderId = getSenderId(msg);
    const senderName = getSenderName(msg);
    const shortId = getShortSenderId(senderId);
    
    // Discover groups
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
    
    // Check for reply to bot message
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
    }
    
    if (contextInfo?.stanzaId) {
        quotedMessageId = contextInfo.stanzaId;
        
        if (isBotMessage(chatId, quotedMessageId)) {
            log('↩️', `Reply to bot from ${senderName} (...${shortId})`);
            await handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName);
            return;
        }
    }
    
    // Handle IMAGES
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
    // Handle PDFs
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        
        if (docMime === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf')) {
            log('📄', `PDF from ${senderName} (...${shortId}): ${fileName}`);
            
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                const caption = msg.message.documentMessage.caption || '';
                
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
        } else {
            log('📎', `Skipping non-PDF: ${fileName}`);
        }
    }
    // Handle AUDIO
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
    // Handle text
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = (msg.message.conversation || msg.message.extendedTextMessage?.text || '').trim();
        
        if (!text) return;
        
        // Trigger: "."
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
                    text: `ℹ️ @${senderId.split('@')[0]}, you have no files buffered.\n\nSend images, PDFs, audio, or text first, then send *.*\n\n💡 _Or reply to my previous response to add context!_`,
                    mentions: [senderId]
                });
            }
        }
        // Help
        else if (text.toLowerCase() === 'help' || text === '?') {
            await sock.sendMessage(chatId, { 
                text: `🏥 *Clinical Profile Bot*\n\n*Supported Content:*\n📷 Images (photos, scans)\n📄 PDFs (reports, documents)\n🎤 Voice messages\n🎵 Audio files\n💬 Text notes & captions\n\n*Basic Usage:*\n1️⃣ Send file(s) and/or text\n2️⃣ Send *.* to process\n\n*👥 Multi-User:*\nEach user's files are tracked separately!\n\n*↩️ Reply Feature:*\nReply to my response to add more context.\n\n*Commands:*\n• *.* - Process YOUR content\n• *clear* - Clear YOUR buffer\n• *status* - Check status` 
            });
        }
        // Clear
        else if (text.toLowerCase() === 'clear') {
            const userItems = clearUserBuffer(chatId, senderId);
            clearUserTimeout(chatId, senderId);
            
            if (userItems.length > 0) {
                const counts = { images: 0, pdfs: 0, audio: 0, texts: 0 };
                userItems.forEach(m => {
                    if (m.type === 'image') counts.images++;
                    else if (m.type === 'pdf') counts.pdfs++;
                    else if (m.type === 'audio' || m.type === 'voice') counts.audio++;
                    else if (m.type === 'text') counts.texts++;
                });
                
                await sock.sendMessage(chatId, { 
                    text: `🗑️ @${senderId.split('@')[0]}, cleared your buffer:\n${counts.images} image(s), ${counts.pdfs} PDF(s), ${counts.audio} audio, ${counts.texts} text(s)`,
                    mentions: [senderId]
                });
            } else {
                await sock.sendMessage(chatId, { 
                    text: `ℹ️ @${senderId.split('@')[0]}, your buffer is empty.`,
                    mentions: [senderId]
                });
            }
        }
        // Status
        else if (text.toLowerCase() === 'status') {
            const stats = getTotalBufferStats(chatId);
            const userCount = getUserBufferCount(chatId, senderId);
            const storedContexts = chatContexts.has(chatId) ? chatContexts.get(chatId).size : 0;
            
            await sock.sendMessage(chatId, { 
                text: `📊 *Status*\n\n*Your Buffer:* ${userCount} item(s)\n\n*Group Total:*\n👥 Active users: ${stats.users}\n📷 Images: ${stats.images}\n📄 PDFs: ${stats.pdfs}\n🎤 Audio: ${stats.audio}\n💬 Texts: ${stats.texts}\n━━━━━━━━━━\n📦 Total buffered: ${stats.total}\n🧠 Stored contexts: ${storedContexts}\n✅ Processed: ${processedCount}\n🗄️ MongoDB: ${mongoConnected ? 'Connected' : 'Not connected'}` 
            });
        }
        // Buffer as text note
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

// ============== HANDLE REPLY TO BOT MESSAGE ==============
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
    
    if (messageType === 'extendedTextMessage') {
        const text = msg.message.extendedTextMessage?.text || '';
        if (text.trim()) {
            newContent.push({
                type: 'text',
                content: text.trim(),
                sender: senderName,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('💬', `Follow-up text from ...${shortId}`);
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
            log('📷', `Follow-up image from ...${shortId}`);
        } catch (error) {
            log('❌', `Image error: ${error.message}`);
        }
    }
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        
        if (docMime === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf')) {
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                newContent.push({
                    type: 'pdf',
                    data: buffer.toString('base64'),
                    mimeType: 'application/pdf',
                    fileName: fileName,
                    caption: msg.message.documentMessage.caption || '',
                    timestamp: Date.now(),
                    isFollowUp: true
                });
                log('📄', `Follow-up PDF from ...${shortId}`);
            } catch (error) {
                log('❌', `PDF error: ${error.message}`);
            }
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
    
    if (newContent.length === 0) {
        await sock.sendMessage(chatId, { 
            text: `ℹ️ @${senderId.split('@')[0]}, please include text, image, PDF, or audio in your reply.`,
            mentions: [senderId]
        });
        return;
    }
    
    const combinedMedia = [...storedContext.mediaFiles, ...newContent];
    
    log('🔄', `Regenerating for ...${shortId}: ${storedContext.mediaFiles.length} original + ${newContent.length} new`);
    
    await processMedia(sock, chatId, combinedMedia, true, storedContext.response, senderId, senderName);
}

// ============== GEMINI ==============
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

async function processMedia(sock, chatId, mediaFiles, isFollowUp = false, previousResponse = null, senderId, senderName) {
    const shortId = getShortSenderId(senderId);
    
    try {
        const counts = { images: 0, pdfs: 0, audio: 0, texts: 0, followUps: 0 };
        const textContents = [];
        const captions = [];
        const binaryMedia = [];
        const followUpTexts = [];
        
        mediaFiles.forEach(m => {
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
            }
            else if (m.type === 'text') {
                counts.texts++;
                if (m.isFollowUp) {
                    followUpTexts.push(`[Additional context from ${m.sender}]: ${m.content}`);
                } else {
                    textContents.push(`[Text note from ${m.sender}]: ${m.content}`);
                }
            }
        });
        
        if (isFollowUp) {
            log('🤖', `Processing follow-up for ...${shortId}: ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.texts} text`);
        } else {
            log('🤖', `Processing for ...${shortId}: ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.texts} text`);
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
        
        const allOriginalText = [...captions, ...textContents];
        let promptText = '';
        
        if (isFollowUp && previousResponse) {
            promptText = `This is a FOLLOW-UP request. The user has provided additional context to refine the Clinical Profile.

=== PREVIOUS CLINICAL PROFILE GENERATED ===
${previousResponse}
=== END PREVIOUS RESPONSE ===

=== ORIGINAL CONTEXT ===
${allOriginalText.length > 0 ? allOriginalText.join('\n\n') : '(Original files are attached below)'}
=== END ORIGINAL CONTEXT ===

=== NEW ADDITIONAL CONTEXT FROM USER ===
${followUpTexts.join('\n\n')}
=== END NEW CONTEXT ===

Please analyze ALL the content (original files + original text + NEW additional context) and generate an UPDATED Clinical Profile that incorporates the new information.`;
            
        } else if (binaryMedia.length > 0 && allOriginalText.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} along with the following additional text notes/context, and generate the Clinical Profile.

=== ADDITIONAL TEXT NOTES ===
${allOriginalText.join('\n\n')}
=== END OF TEXT NOTES ===

For audio files, transcribe the content first, then extract medical information.`;
        } 
        else if (binaryMedia.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} containing medical information and generate the Clinical Profile. For audio files, transcribe the content first.`;
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
        
        const result = await model.generateContent(requestContent);
        const clinicalProfile = result.response.text();
        
        log('✅', `Done for ...${shortId}!`);
        processedCount++;
        
        console.log('\n' + '═'.repeat(60));
        console.log(`👤 User: ${senderName} (...${shortId})`);
        if (isFollowUp) console.log(`🔄 FOLLOW-UP RESPONSE`);
        console.log(`📊 ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.texts} text`);
        console.log(`⏰ ${new Date().toLocaleString()}`);
        console.log('═'.repeat(60));
        console.log(clinicalProfile);
        console.log('═'.repeat(60) + '\n');
        
        // Typing indicator
        await sock.sendPresenceUpdate('composing', chatId);
        const delay = Math.floor(Math.random() * (CONFIG.TYPING_DELAY_MAX - CONFIG.TYPING_DELAY_MIN)) + CONFIG.TYPING_DELAY_MIN;
        await new Promise(resolve => setTimeout(resolve, delay));
        await sock.sendPresenceUpdate('paused', chatId);
        
        // Send response with mention
        let sentMessage;
        const responseText = clinicalProfile.length <= 3800 
            ? clinicalProfile 
            : clinicalProfile.substring(0, 3800) + '\n\n_(truncated)_';
        
        sentMessage = await sock.sendMessage(chatId, { 
            text: responseText,
            mentions: [senderId]
        });
        
        // Store context for future replies
        if (sentMessage?.key?.id) {
            const messageId = sentMessage.key.id;
            trackBotMessage(chatId, messageId);
            storeContext(chatId, messageId, mediaFiles, clinicalProfile, senderId);
            log('💾', `Context stored for ...${shortId}`);
        }
        
        log('📤', `Sent to ...${shortId}!`);
        
    } catch (error) {
        log('❌', `Error for ...${shortId}: ${error.message}`);
        console.error(error);
        
        await sock.sendPresenceUpdate('composing', chatId);
        await new Promise(r => setTimeout(r, 1500));
        
        await sock.sendMessage(chatId, { 
            text: `❌ @${senderId.split('@')[0]}, error processing your request:\n_${error.message}_\n\nPlease try again.`,
            mentions: [senderId]
        });
    }
}

// ============== START ==============
console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║         WhatsApp Clinical Profile Bot v2.0                 ║');
console.log('║                                                            ║');
console.log('║  📷 Images  📄 PDFs  🎤 Voice  🎵 Audio  💬 Text           ║');
console.log('║                                                            ║');
console.log('║  ✨ Per-User Buffers - Each user processed separately     ║');
console.log('║  ↩️ Reply to bot response to add context                   ║');
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

// Start the bot
(async () => {
    try {
        // Connect MongoDB first
        await connectMongoDB();
        
        // Then start bot
        await startBot();
    } catch (error) {
        log('💥', `Startup error: ${error.message}`);
        console.error(error);
        process.exit(1);
    }
})();import { GoogleGenerativeAI } from '@google/generative-ai';
import pino from 'pino';
import QRCode from 'qrcode';
import express from 'express';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import mongoose from 'mongoose';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============== CONFIGURATION ==============
const CONFIG = {
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    GEMINI_MODEL: 'gemini-2.0-flash',
    MONGODB_URI: process.env.MONGODB_URI,
    ALLOWED_GROUP_ID: process.env.ALLOWED_GROUP_ID || '',
    MEDIA_TIMEOUT_MS: 300000, // 5 minutes
    CONTEXT_RETENTION_MS: 1800000, // 30 minutes
    MAX_STORED_CONTEXTS: 20,
    TRIGGER_TEXT: '.',
    COMMANDS: ['.', 'help', '?', 'clear', 'status'],
    TYPING_DELAY_MIN: 3000,
    TYPING_DELAY_MAX: 6000,
    SYSTEM_INSTRUCTION: `You are an expert medical AI assistant specializing in radiology. You have to extract transcript / raw text from one or more uploaded files (images, PDFs, or audio recordings). You may also receive additional text context provided by the user. Your task is to analyze all content to create a concise and comprehensive "Clinical Profile".

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT

FOR AUDIO FILES: Transcribe the audio content carefully and extract all relevant medical information mentioned.

FOR TEXT MESSAGES: These may contain additional clinical context, patient history, or notes that should be incorporated into the Clinical Profile.

FOR FOLLOW-UP REQUESTS: If the user provides additional context or corrections after receiving a Clinical Profile, incorporate that new information and regenerate an updated Clinical Profile.

YOUR RESPONSE MUST BE BASED SOLELY ON THE PROVIDED CONTENT (files AND text).

Follow these strict instructions:

Analyze All Content: Meticulously examine all provided files - images, PDFs, and audio recordings, as well as any accompanying text messages. This may include prior medical scan reports (like USG, CT, MRI), clinical notes, voice memos, or other relevant documents.

Extract Key Information: From the content, identify and extract all pertinent information, such as:

Scan types (e.g., USG, CT Brain).

Dates of scans or documents.

Key findings, measurements, or impressions from reports.

Relevant clinical history mentioned in notes, audio, or text messages.

Synthesize into a Clinical Profile:

Combine all extracted information into a single, cohesive paragraph. This represents a 100% recreation of the relevant clinical details from the provided content.

If there are repeated or vague findings across multiple documents, synthesize them into a single, concise statement.

Frame sentences properly to be concise, but you MUST NOT omit any important clinical details. Prioritize completeness of clinical information over extreme brevity.

You MUST strictly exclude any mention of the patient's name, age, or gender.

If multiple dated scan reports are present, you MUST arrange their summaries chronologically in ascending order based on their dates.

If a date is not available for a scan, refer to it as "Previous [Scan Type]...".

Formatting:

The final output MUST be a single paragraph.

This paragraph MUST start with "Clinical Profile:" and the entire content (including the prefix) must be wrapped in single asterisks. For example: "*Clinical Profile: Previous USG dated 01/01/2023 showed mild hepatomegaly. Patient also has a H/o hypertension as noted in the clinical sheet.*"

Output:

Do not output the raw transcribed text.

Do not output JSON or Markdown code blocks.

Return ONLY the single formatted paragraph described above.

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT`
};

// ============== MONGODB SETUP ==============
const sessionSchema = new mongoose.Schema({
    key: { type: String, required: true, unique: true },
    value: { type: mongoose.Schema.Types.Mixed, required: true },
    updatedAt: { type: Date, default: Date.now }
}, { collection: 'whatsapp_sessions' });

sessionSchema.index({ updatedAt: 1 }, { expireAfterSeconds: 86400 * 30 }); // 30 days TTL

let SessionModel;

// ============== MONGODB AUTH STATE ==============
async function useMongoDBAuthState() {
    // Ensure model is created
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

    // Load existing creds or create new
    let creds = await readData('auth_creds');
    
    if (!creds) {
        // Import and initialize fresh credentials
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

// ============== SETUP ==============
// Per-User Media Buffers: chatId -> senderId -> [media files]
const chatMediaBuffers = new Map();
// Per-User Timeouts: chatId -> senderId -> timeout
const chatTimeouts = new Map();
// Discovered groups
const discoveredGroups = new Map();
// Context storage for reply feature: chatId -> Map(messageId -> {mediaFiles, response, timestamp, senderId})
const chatContexts = new Map();
// Track bot's own message IDs: chatId -> Set of message IDs
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

// Get sender ID from message
function getSenderId(msg) {
    return msg.key.participant || msg.key.remoteJid;
}

// Get sender name (phone number in readable format)
function getSenderName(msg) {
    const senderId = getSenderId(msg);
    // Extract phone number from JID
    const phone = senderId.split('@')[0];
    return msg.pushName || phone;
}

// Get short sender ID for logging
function getShortSenderId(senderId) {
    const phone = senderId.split('@')[0];
    if (phone.length > 6) {
        return phone.slice(-4); // Last 4 digits
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

// ============== PER-USER BUFFER MANAGEMENT ==============
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
    const stats = { users: 0, images: 0, pdfs: 0, audio: 0, texts: 0, total: 0 };
    if (!chatMediaBuffers.has(chatId)) return stats;
    
    const chatBuffer = chatMediaBuffers.get(chatId);
    stats.users = chatBuffer.size;
    
    for (const [senderId, items] of chatBuffer) {
        items.forEach(m => {
            if (m.type === 'image') stats.images++;
            else if (m.type === 'pdf') stats.pdfs++;
            else if (m.type === 'audio' || m.type === 'voice') stats.audio++;
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
    
    // Clear existing timeout for this user
    if (chatTimeoutMap.has(senderId)) {
        clearTimeout(chatTimeoutMap.get(senderId));
    }
    
    // Set new timeout
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

// ============== CONTEXT MANAGEMENT ==============
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

// ============== WEB SERVER ==============
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
                max-width: 550px;
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
                gap: 8px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            .stat {
                background: rgba(255,255,255,0.1);
                padding: 10px 12px;
                border-radius: 10px;
                min-width: 55px;
            }
            .stat-value { font-size: 18px; font-weight: bold; }
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
                    <div class="stat"><div class="stat-value">${stats.texts}</div><div class="stat-label">💬 Texts</div></div>
                    <div class="stat"><div class="stat-value">${storedContextsCount}</div><div class="stat-label">🧠 Contexts</div></div>
                    <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">✅ Done</div></div>
                </div>
                <div class="info-box">
                    <h3>✨ Usage:</h3>
                    <p><strong>New Request:</strong><br>
                    1. Send files/text → Send <strong>.</strong> → Get profile<br><br>
                    <strong>👥 Multi-User Support:</strong><br>
                    Each user's files are processed separately!<br><br>
                    <strong>↩️ Reply Feature:</strong><br>
                    Reply to bot's response to add more context!</p>
                </div>
                <div class="media-support">
                    <span class="feature-badge">📷 Images</span>
                    <span class="feature-badge">📄 PDFs</span>
                    <span class="feature-badge">🎤 Voice</span>
                    <span class="feature-badge">💬 Text</span>
                    <span class="feature-badge">👥 Per-User</span>
                    <span class="feature-badge">🗄️ Persistent</span>
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

// ============== LOAD BAILEYS ==============
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

// ============== MONGODB CONNECTION ==============
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

// ============== WHATSAPP BOT ==============
async function startBot() {
    try {
        botStatus = 'Initializing...';
        log('🚀', 'Starting WhatsApp Bot...');
        
        if (!makeWASocket) await loadBaileys();
        
        // Get auth state
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
            // Fallback to file auth (will lose session on restart)
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
        
        // Store auth state for later
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

        // Connection updates
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
                
                // Check if logged out
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
                    // Reconnect for other errors
                    const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
                    log('🔄', `Reconnecting in 5 seconds... (shouldReconnect: ${shouldReconnect})`);
                    setTimeout(startBot, 5000);
                }
                
            } else if (connection === 'open') {
                isConnected = true;
                qrCodeDataURL = null;
                botStatus = 'Connected';
                
                log('✅', '🎉 CONNECTED TO WHATSAPP!');
                
                // Save credentials immediately
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

        // Credentials update
        sock.ev.on('creds.update', async () => {
            if (authState?.saveCreds) {
                await authState.saveCreds();
            }
        });

        // Messages
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

// ============== MESSAGE HANDLER ==============
async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    
    if (chatId === 'status@broadcast') return;
    
    const isGroup = chatId.endsWith('@g.us');
    const senderId = getSenderId(msg);
    const senderName = getSenderName(msg);
    const shortId = getShortSenderId(senderId);
    
    // Discover groups
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
    
    // Check for reply to bot message
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
    }
    
    if (contextInfo?.stanzaId) {
        quotedMessageId = contextInfo.stanzaId;
        
        if (isBotMessage(chatId, quotedMessageId)) {
            log('↩️', `Reply to bot from ${senderName} (...${shortId})`);
            await handleReplyToBot(sock, msg, chatId, quotedMessageId, senderId, senderName);
            return;
        }
    }
    
    // Handle IMAGES
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
    // Handle PDFs
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        
        if (docMime === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf')) {
            log('📄', `PDF from ${senderName} (...${shortId}): ${fileName}`);
            
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                const caption = msg.message.documentMessage.caption || '';
                
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
        } else {
            log('📎', `Skipping non-PDF: ${fileName}`);
        }
    }
    // Handle AUDIO
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
    // Handle text
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = (msg.message.conversation || msg.message.extendedTextMessage?.text || '').trim();
        
        if (!text) return;
        
        // Trigger: "."
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
                    text: `ℹ️ @${senderId.split('@')[0]}, you have no files buffered.\n\nSend images, PDFs, audio, or text first, then send *.*\n\n💡 _Or reply to my previous response to add context!_`,
                    mentions: [senderId]
                });
            }
        }
        // Help
        else if (text.toLowerCase() === 'help' || text === '?') {
            await sock.sendMessage(chatId, { 
                text: `🏥 *Clinical Profile Bot*\n\n*Supported Content:*\n📷 Images (photos, scans)\n📄 PDFs (reports, documents)\n🎤 Voice messages\n🎵 Audio files\n💬 Text notes & captions\n\n*Basic Usage:*\n1️⃣ Send file(s) and/or text\n2️⃣ Send *.* to process\n\n*👥 Multi-User:*\nEach user's files are tracked separately!\n\n*↩️ Reply Feature:*\nReply to my response to add more context.\n\n*Commands:*\n• *.* - Process YOUR content\n• *clear* - Clear YOUR buffer\n• *status* - Check status` 
            });
        }
        // Clear
        else if (text.toLowerCase() === 'clear') {
            const userItems = clearUserBuffer(chatId, senderId);
            clearUserTimeout(chatId, senderId);
            
            if (userItems.length > 0) {
                const counts = { images: 0, pdfs: 0, audio: 0, texts: 0 };
                userItems.forEach(m => {
                    if (m.type === 'image') counts.images++;
                    else if (m.type === 'pdf') counts.pdfs++;
                    else if (m.type === 'audio' || m.type === 'voice') counts.audio++;
                    else if (m.type === 'text') counts.texts++;
                });
                
                await sock.sendMessage(chatId, { 
                    text: `🗑️ @${senderId.split('@')[0]}, cleared your buffer:\n${counts.images} image(s), ${counts.pdfs} PDF(s), ${counts.audio} audio, ${counts.texts} text(s)`,
                    mentions: [senderId]
                });
            } else {
                await sock.sendMessage(chatId, { 
                    text: `ℹ️ @${senderId.split('@')[0]}, your buffer is empty.`,
                    mentions: [senderId]
                });
            }
        }
        // Status
        else if (text.toLowerCase() === 'status') {
            const stats = getTotalBufferStats(chatId);
            const userCount = getUserBufferCount(chatId, senderId);
            const storedContexts = chatContexts.has(chatId) ? chatContexts.get(chatId).size : 0;
            
            await sock.sendMessage(chatId, { 
                text: `📊 *Status*\n\n*Your Buffer:* ${userCount} item(s)\n\n*Group Total:*\n👥 Active users: ${stats.users}\n📷 Images: ${stats.images}\n📄 PDFs: ${stats.pdfs}\n🎤 Audio: ${stats.audio}\n💬 Texts: ${stats.texts}\n━━━━━━━━━━\n📦 Total buffered: ${stats.total}\n🧠 Stored contexts: ${storedContexts}\n✅ Processed: ${processedCount}\n🗄️ MongoDB: ${mongoConnected ? 'Connected' : 'Not connected'}` 
            });
        }
        // Buffer as text note
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

// ============== HANDLE REPLY TO BOT MESSAGE ==============
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
    
    if (messageType === 'extendedTextMessage') {
        const text = msg.message.extendedTextMessage?.text || '';
        if (text.trim()) {
            newContent.push({
                type: 'text',
                content: text.trim(),
                sender: senderName,
                timestamp: Date.now(),
                isFollowUp: true
            });
            log('💬', `Follow-up text from ...${shortId}`);
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
            log('📷', `Follow-up image from ...${shortId}`);
        } catch (error) {
            log('❌', `Image error: ${error.message}`);
        }
    }
    else if (messageType === 'documentMessage') {
        const docMime = msg.message.documentMessage.mimetype || '';
        const fileName = msg.message.documentMessage.fileName || 'document';
        
        if (docMime === 'application/pdf' || fileName.toLowerCase().endsWith('.pdf')) {
            try {
                const buffer = await downloadMediaMessage(msg, 'buffer', {});
                
                newContent.push({
                    type: 'pdf',
                    data: buffer.toString('base64'),
                    mimeType: 'application/pdf',
                    fileName: fileName,
                    caption: msg.message.documentMessage.caption || '',
                    timestamp: Date.now(),
                    isFollowUp: true
                });
                log('📄', `Follow-up PDF from ...${shortId}`);
            } catch (error) {
                log('❌', `PDF error: ${error.message}`);
            }
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
    
    if (newContent.length === 0) {
        await sock.sendMessage(chatId, { 
            text: `ℹ️ @${senderId.split('@')[0]}, please include text, image, PDF, or audio in your reply.`,
            mentions: [senderId]
        });
        return;
    }
    
    const combinedMedia = [...storedContext.mediaFiles, ...newContent];
    
    log('🔄', `Regenerating for ...${shortId}: ${storedContext.mediaFiles.length} original + ${newContent.length} new`);
    
    await processMedia(sock, chatId, combinedMedia, true, storedContext.response, senderId, senderName);
}

// ============== GEMINI ==============
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

async function processMedia(sock, chatId, mediaFiles, isFollowUp = false, previousResponse = null, senderId, senderName) {
    const shortId = getShortSenderId(senderId);
    
    try {
        const counts = { images: 0, pdfs: 0, audio: 0, texts: 0, followUps: 0 };
        const textContents = [];
        const captions = [];
        const binaryMedia = [];
        const followUpTexts = [];
        
        mediaFiles.forEach(m => {
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
            }
            else if (m.type === 'text') {
                counts.texts++;
                if (m.isFollowUp) {
                    followUpTexts.push(`[Additional context from ${m.sender}]: ${m.content}`);
                } else {
                    textContents.push(`[Text note from ${m.sender}]: ${m.content}`);
                }
            }
        });
        
        if (isFollowUp) {
            log('🤖', `Processing follow-up for ...${shortId}: ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.texts} text`);
        } else {
            log('🤖', `Processing for ...${shortId}: ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.texts} text`);
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
        
        const allOriginalText = [...captions, ...textContents];
        let promptText = '';
        
        if (isFollowUp && previousResponse) {
            promptText = `This is a FOLLOW-UP request. The user has provided additional context to refine the Clinical Profile.

=== PREVIOUS CLINICAL PROFILE GENERATED ===
${previousResponse}
=== END PREVIOUS RESPONSE ===

=== ORIGINAL CONTEXT ===
${allOriginalText.length > 0 ? allOriginalText.join('\n\n') : '(Original files are attached below)'}
=== END ORIGINAL CONTEXT ===

=== NEW ADDITIONAL CONTEXT FROM USER ===
${followUpTexts.join('\n\n')}
=== END NEW CONTEXT ===

Please analyze ALL the content (original files + original text + NEW additional context) and generate an UPDATED Clinical Profile that incorporates the new information.`;
            
        } else if (binaryMedia.length > 0 && allOriginalText.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} along with the following additional text notes/context, and generate the Clinical Profile.

=== ADDITIONAL TEXT NOTES ===
${allOriginalText.join('\n\n')}
=== END OF TEXT NOTES ===

For audio files, transcribe the content first, then extract medical information.`;
        } 
        else if (binaryMedia.length > 0) {
            promptText = `Analyze these ${promptParts.join(', ')} containing medical information and generate the Clinical Profile. For audio files, transcribe the content first.`;
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
        
        const result = await model.generateContent(requestContent);
        const clinicalProfile = result.response.text();
        
        log('✅', `Done for ...${shortId}!`);
        processedCount++;
        
        console.log('\n' + '═'.repeat(60));
        console.log(`👤 User: ${senderName} (...${shortId})`);
        if (isFollowUp) console.log(`🔄 FOLLOW-UP RESPONSE`);
        console.log(`📊 ${counts.images} img, ${counts.pdfs} PDF, ${counts.audio} audio, ${counts.texts} text`);
        console.log(`⏰ ${new Date().toLocaleString()}`);
        console.log('═'.repeat(60));
        console.log(clinicalProfile);
        console.log('═'.repeat(60) + '\n');
        
        // Typing indicator
        await sock.sendPresenceUpdate('composing', chatId);
        const delay = Math.floor(Math.random() * (CONFIG.TYPING_DELAY_MAX - CONFIG.TYPING_DELAY_MIN)) + CONFIG.TYPING_DELAY_MIN;
        await new Promise(resolve => setTimeout(resolve, delay));
        await sock.sendPresenceUpdate('paused', chatId);
        
        // Send response with mention
        let sentMessage;
        const responseText = clinicalProfile.length <= 3800 
            ? clinicalProfile 
            : clinicalProfile.substring(0, 3800) + '\n\n_(truncated)_';
        
        sentMessage = await sock.sendMessage(chatId, { 
            text: responseText,
            mentions: [senderId]
        });
        
        // Store context for future replies
        if (sentMessage?.key?.id) {
            const messageId = sentMessage.key.id;
            trackBotMessage(chatId, messageId);
            storeContext(chatId, messageId, mediaFiles, clinicalProfile, senderId);
            log('💾', `Context stored for ...${shortId}`);
        }
        
        log('📤', `Sent to ...${shortId}!`);
        
    } catch (error) {
        log('❌', `Error for ...${shortId}: ${error.message}`);
        console.error(error);
        
        await sock.sendPresenceUpdate('composing', chatId);
        await new Promise(r => setTimeout(r, 1500));
        
        await sock.sendMessage(chatId, { 
            text: `❌ @${senderId.split('@')[0]}, error processing your request:\n_${error.message}_\n\nPlease try again.`,
            mentions: [senderId]
        });
    }
}

// ============== START ==============
console.log('\n╔════════════════════════════════════════════════════════════╗');
console.log('║         WhatsApp Clinical Profile Bot v2.0                 ║');
console.log('║                                                            ║');
console.log('║  📷 Images  📄 PDFs  🎤 Voice  🎵 Audio  💬 Text           ║');
console.log('║                                                            ║');
console.log('║  ✨ Per-User Buffers - Each user processed separately     ║');
console.log('║  ↩️ Reply to bot response to add context                   ║');
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

// Start the bot
(async () => {
    try {
        // Connect MongoDB first
        await connectMongoDB();
        
        // Then start bot
        await startBot();
    } catch (error) {
        log('💥', `Startup error: ${error.message}`);
        console.error(error);
        process.exit(1);
    }
})();
