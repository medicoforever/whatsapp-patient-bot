import { GoogleGenerativeAI } from '@google/generative-ai';
import pino from 'pino';
import QRCode from 'qrcode';
import express from 'express';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============== CONFIGURATION ==============
const CONFIG = {
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    GEMINI_MODEL: 'gemini-3-flash-preview',
    
    // SET THIS TO YOUR GROUP ID (from environment variable)
    // Leave empty to see group IDs in logs first
    ALLOWED_GROUP_ID: process.env.ALLOWED_GROUP_ID || '',
    
    // Time before auto-clearing images (5 minutes)
    IMAGE_TIMEOUT_MS: 300000,
    
    // Trigger to process images
    TRIGGER_TEXT: '.',
    
    SYSTEM_INSTRUCTION: `You are an expert medical AI assistant specializing in radiology. You have to extract transcript / raw text from one or more uploaded pictures. Your task is to analyze that text to create a concise and comprehensive "Clinical Profile".

IMPORTANT INSTRUCTION - IF THE HANDWRITTEN TEXT IS NOT LEGIBLE, FEEL FREE TO USE CODE INTERPRETATION AND LOGIC IN THE CONTEXT OF OTHER TEXTS TO DECIPHER THE ILLEGIBLE TEXT

YOUR RESPONSE MUST BE BASED SOLELY ON THE PROVIDED TRANSCRIPTIONS.

Follow these strict instructions:

Analyze All Transcriptions: Meticulously examine all provided text. This may include prior medical scan reports (like USG, CT, MRI), clinical notes, or other relevant documents.

Extract Key Information: From the text, identify and extract all pertinent information, such as:

Scan types (e.g., USG, CT Brain).

Dates of scans or documents.

Key findings, measurements, or impressions from reports.

Relevant clinical history mentioned in notes.

Synthesize into a Clinical Profile:

Combine all extracted information into a single, cohesive paragraph. This represents a 100% recreation of the relevant clinical details from the provided text.

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

// ============== SETUP ==============
const chatImageBuffers = new Map();
const chatTimeouts = new Map();
const discoveredGroups = new Map();

let sock = null;
let isConnected = false;
let qrCodeDataURL = null;
let processedCount = 0;
let botStatus = 'Starting...';
let lastError = null;

let makeWASocket, useMultiFileAuthState, DisconnectReason, downloadMediaMessage, fetchLatestBaileysVersion;

function log(emoji, message) {
    const time = new Date().toLocaleTimeString();
    console.log(`[${time}] ${emoji} ${message}`);
}

// Check if chat is the allowed group
function isAllowedGroup(chatId) {
    // If no group is configured, we're in "discovery mode"
    if (!CONFIG.ALLOWED_GROUP_ID) {
        return false;
    }
    return chatId === CONFIG.ALLOWED_GROUP_ID;
}

// ============== WEB SERVER ==============
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    const imageCount = chatImageBuffers.has(CONFIG.ALLOWED_GROUP_ID) 
        ? chatImageBuffers.get(CONFIG.ALLOWED_GROUP_ID).length 
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
                gap: 15px;
                margin-top: 20px;
            }
            .stat {
                background: rgba(255,255,255,0.1);
                padding: 10px 15px;
                border-radius: 10px;
            }
            .stat-value { font-size: 22px; font-weight: bold; }
            .stat-label { font-size: 10px; opacity: 0.8; }
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 WhatsApp Patient Bot</h1>
            <p class="subtitle">Medical Image Clinical Profile Generator</p>
    `;
    
    if (isConnected) {
        if (CONFIG.ALLOWED_GROUP_ID) {
            // Configured - show active status
            const groupName = discoveredGroups.get(CONFIG.ALLOWED_GROUP_ID) || 'Configured Group';
            html += `
                <div class="status connected">✅ ACTIVE IN GROUP</div>
                <div class="configured">
                    <strong>🎯 Active Group:</strong> ${groupName}<br>
                    <code style="font-size:10px">${CONFIG.ALLOWED_GROUP_ID}</code>
                </div>
                <div class="stats">
                    <div class="stat"><div class="stat-value">${imageCount}</div><div class="stat-label">Buffered</div></div>
                    <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">Processed</div></div>
                </div>
                <div class="info-box">
                    <h3>✨ Usage:</h3>
                    <p>1. Send image(s) in the group<br>
                    2. Send <strong>.</strong> to process<br>
                    3. Get Clinical Profile!</p>
                </div>
            `;
        } else {
            // Not configured - show discovery mode
            html += `
                <div class="status warning">⚠️ DISCOVERY MODE</div>
                <div class="not-configured">
                    <strong>No group configured yet!</strong><br>
                    Send a message in your target group to discover its ID.
                </div>
            `;
            
            if (discoveredGroups.size > 0) {
                html += `<div class="group-list"><h4>📋 Discovered Groups (send message to discover):</h4>`;
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
            } else {
                html += `
                    <div class="info-box">
                        <h3>👋 Getting Started:</h3>
                        <p>Send any message in your target WhatsApp group.<br>
                        The group ID will appear here.</p>
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
    res.json({ 
        status: 'running',
        connected: isConnected,
        configuredGroup: CONFIG.ALLOWED_GROUP_ID || 'NOT SET',
        discoveredGroups: Object.fromEntries(discoveredGroups),
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
        useMultiFileAuthState = baileys.useMultiFileAuthState;
        DisconnectReason = baileys.DisconnectReason;
        downloadMediaMessage = baileys.downloadMediaMessage;
        fetchLatestBaileysVersion = baileys.fetchLatestBaileysVersion;
        
        log('✅', 'Baileys loaded!');
        return true;
    } catch (error) {
        log('❌', `Failed: ${error.message}`);
        throw error;
    }
}

// ============== WHATSAPP BOT ==============
async function startBot() {
    try {
        botStatus = 'Initializing...';
        log('🚀', 'Starting WhatsApp Bot...');
        
        if (!makeWASocket) await loadBaileys();
        
        const authPath = join(__dirname, 'auth_session');
        const { state, saveCreds } = await useMultiFileAuthState(authPath);
        
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
            browser: ['Ubuntu', 'Chrome', '20.0.04'],
            markOnlineOnConnect: false,
            getMessage: async () => ({ conversation: '' })
        });

        sock.ev.on('connection.update', async (update) => {
            const { connection, lastDisconnect, qr } = update;
            
            if (qr) {
                try {
                    botStatus = 'QR Code ready';
                    qrCodeDataURL = await QRCode.toDataURL(qr, {
                        width: 300, margin: 2,
                        color: { dark: '#128C7E', light: '#FFFFFF' }
                    });
                    isConnected = false;
                    log('📱', 'QR Code ready!');
                } catch (err) {
                    lastError = err.message;
                }
            }
            
            if (connection === 'close') {
                isConnected = false;
                qrCodeDataURL = null;
                const code = lastDisconnect?.error?.output?.statusCode;
                
                if (code === 401 || code === 405 || code === DisconnectReason?.loggedOut) {
                    try { fs.rmSync(authPath, { recursive: true, force: true }); } catch (e) {}
                }
                
                setTimeout(startBot, 5000);
                
            } else if (connection === 'open') {
                isConnected = true;
                qrCodeDataURL = null;
                log('✅', '🎉 CONNECTED!');
                
                if (CONFIG.ALLOWED_GROUP_ID) {
                    log('🎯', `Bot active ONLY in: ${CONFIG.ALLOWED_GROUP_ID}`);
                } else {
                    log('⚠️', 'No group configured! Send a message in target group to get its ID.');
                }
            }
        });

        sock.ev.on('creds.update', saveCreds);

        sock.ev.on('messages.upsert', async ({ messages }) => {
            for (const msg of messages) {
                if (msg.key.fromMe) continue;
                if (!msg.message) continue;
                await handleMessage(sock, msg);
            }
        });
        
    } catch (error) {
        log('💥', `Error: ${error.message}`);
        setTimeout(startBot, 10000);
    }
}

// ============== MESSAGE HANDLER ==============
async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    
    // Skip status broadcasts
    if (chatId === 'status@broadcast') return;
    
    const isGroup = chatId.endsWith('@g.us');
    
    // If it's a group, discover/track it
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
    
    // ========== KEY LOGIC ==========
    // If group is NOT configured OR this is NOT the allowed group → IGNORE
    if (!isAllowedGroup(chatId)) {
        // Only log group discoveries, completely ignore the message
        return;
    }
    // ===============================
    
    // From here, we're in the ALLOWED GROUP only
    const messageType = Object.keys(msg.message)[0];
    const senderName = msg.pushName || 'Unknown';
    
    // Handle images
    if (messageType === 'imageMessage') {
        log('📷', `Image from ${senderName}`);
        
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            
            if (!chatImageBuffers.has(chatId)) {
                chatImageBuffers.set(chatId, []);
            }
            
            chatImageBuffers.get(chatId).push({
                data: buffer.toString('base64'),
                mimeType: msg.message.imageMessage.mimetype || 'image/jpeg',
                timestamp: Date.now()
            });
            
            const count = chatImageBuffers.get(chatId).length;
            log('📦', `Buffer: ${count} image(s)`);
            
            // Reset timeout
            if (chatTimeouts.has(chatId)) {
                clearTimeout(chatTimeouts.get(chatId));
            }
            
            chatTimeouts.set(chatId, setTimeout(() => {
                if (chatImageBuffers.has(chatId)) {
                    log('⏰', 'Auto-clearing images');
                    chatImageBuffers.delete(chatId);
                    chatTimeouts.delete(chatId);
                }
            }, CONFIG.IMAGE_TIMEOUT_MS));
            
            // Acknowledge
            if (count === 1) {
                await sock.sendMessage(chatId, { 
                    text: `📷 *${count} image received*\n\n_Send more if needed, then send *.*  to process._` 
                });
            } else if (count % 5 === 0 || count === 2) {
                await sock.sendMessage(chatId, { 
                    text: `📷 *${count} images received*\n_Send *.* when ready._` 
                });
            }
            
        } catch (error) {
            log('❌', `Error: ${error.message}`);
        }
    }
    // Handle text
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = (msg.message.conversation || msg.message.extendedTextMessage?.text || '').trim();
        
        // Trigger: "."
        if (text === CONFIG.TRIGGER_TEXT) {
            log('🔔', `Trigger from ${senderName}`);
            
            if (chatImageBuffers.has(chatId) && chatImageBuffers.get(chatId).length > 0) {
                if (chatTimeouts.has(chatId)) {
                    clearTimeout(chatTimeouts.get(chatId));
                    chatTimeouts.delete(chatId);
                }
                
                const images = chatImageBuffers.get(chatId);
                chatImageBuffers.delete(chatId);
                
                await processImages(sock, chatId, images);
            } else {
                await sock.sendMessage(chatId, { 
                    text: `❌ *No images to process*\n\n_Send image(s) first, then send *.*_` 
                });
            }
        }
        // Help
        else if (text.toLowerCase() === 'help' || text === '?') {
            await sock.sendMessage(chatId, { 
                text: `🏥 *Clinical Profile Bot*\n\n*Usage:*\n1️⃣ Send medical image(s)\n2️⃣ Send *.* to process\n\n*Commands:*\n• *.* - Process images\n• *clear* - Clear buffer\n• *status* - Check status` 
            });
        }
        // Clear
        else if (text.toLowerCase() === 'clear') {
            if (chatImageBuffers.has(chatId)) {
                const count = chatImageBuffers.get(chatId).length;
                chatImageBuffers.delete(chatId);
                if (chatTimeouts.has(chatId)) {
                    clearTimeout(chatTimeouts.get(chatId));
                    chatTimeouts.delete(chatId);
                }
                await sock.sendMessage(chatId, { text: `🗑️ Cleared ${count} image(s)` });
            } else {
                await sock.sendMessage(chatId, { text: `ℹ️ No images buffered` });
            }
        }
        // Status
        else if (text.toLowerCase() === 'status') {
            const count = chatImageBuffers.has(chatId) ? chatImageBuffers.get(chatId).length : 0;
            await sock.sendMessage(chatId, { 
                text: `📊 *Status*\n📷 Buffered: ${count}\n✅ Processed: ${processedCount}` 
            });
        }
    }
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
    log('❌', `Gemini error: ${error.message}`);
}

async function processImages(sock, chatId, images) {
    try {
        log('🤖', `Processing ${images.length} images...`);
        
        await sock.sendMessage(chatId, { 
            text: `⏳ *Processing ${images.length} image(s)...*\n_Please wait..._` 
        });
        
        const imageParts = images.map(img => ({
            inlineData: { data: img.data, mimeType: img.mimeType }
        }));
        
        const result = await model.generateContent([
            `Analyze these ${images.length} medical document image(s) and generate the Clinical Profile.`,
            ...imageParts
        ]);
        
        const clinicalProfile = result.response.text();
        
        log('✅', 'Done!');
        processedCount++;
        
        // Log to console
        console.log('\n' + '═'.repeat(50));
        console.log(`📸 ${images.length} images | ⏰ ${new Date().toLocaleString()}`);
        console.log('═'.repeat(50));
        console.log(clinicalProfile);
        console.log('═'.repeat(50) + '\n');
        
        // Send response (handle long messages)
        if (clinicalProfile.length <= 4000) {
            await sock.sendMessage(chatId, { text: clinicalProfile });
        } else {
            const chunks = clinicalProfile.match(/.{1,3900}/gs) || [];
            for (let i = 0; i < chunks.length; i++) {
                if (i > 0) await new Promise(r => setTimeout(r, 1000));
                await sock.sendMessage(chatId, { 
                    text: chunks[i] + (i < chunks.length - 1 ? '\n\n_(continued...)_' : '')
                });
            }
        }
        
        log('📤', 'Sent!');
        
    } catch (error) {
        log('❌', `Error: ${error.message}`);
        await sock.sendMessage(chatId, { 
            text: `❌ *Error*\n_${error.message}_\n\nPlease try again.` 
        });
    }
}

// ============== START ==============
console.log('\n╔═══════════════════════════════════════════════════════╗');
console.log('║      WhatsApp Clinical Profile Bot                    ║');
console.log('║                                                       ║');
console.log('║  🔒 Works ONLY in ONE specific group                  ║');
console.log('║  💬 Normal WhatsApp everywhere else                   ║');
console.log('╚═══════════════════════════════════════════════════════╝\n');

log('🏁', 'Starting...');

if (CONFIG.ALLOWED_GROUP_ID) {
    log('🎯', `Configured for group: ${CONFIG.ALLOWED_GROUP_ID}`);
} else {
    log('⚠️', 'No group configured! Bot will discover groups when you message them.');
    log('💡', 'After finding your group ID, add ALLOWED_GROUP_ID to Render environment.');
}

startBot();
