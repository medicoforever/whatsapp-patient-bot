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
    
    // Time to wait before auto-clearing images (5 minutes)
    IMAGE_TIMEOUT_MS: 300000,
    
    // Trigger text to process images
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
// Store images per chat (chatId -> array of images)
const chatImageBuffers = new Map();
const chatTimeouts = new Map();

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

// ============== WEB SERVER ==============
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    const totalBuffered = Array.from(chatImageBuffers.values()).reduce((sum, arr) => sum + arr.length, 0);
    
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
                max-width: 450px;
                width: 100%;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 { margin: 0 0 10px 0; font-size: 24px; }
            .subtitle { opacity: 0.9; margin-bottom: 20px; font-size: 14px; }
            .model-badge {
                background: rgba(255,255,255,0.2);
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 11px;
                display: inline-block;
                margin-bottom: 15px;
            }
            .status {
                padding: 15px 20px;
                border-radius: 12px;
                margin: 20px 0;
                font-size: 16px;
                font-weight: 600;
            }
            .connected { background: #4CAF50; animation: pulse 2s infinite; }
            .waiting { background: rgba(255,255,255,0.2); }
            .error { background: #f44336; }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }
            .qr-container {
                background: white;
                padding: 15px;
                border-radius: 15px;
                display: inline-block;
                margin: 20px 0;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }
            .qr-container img {
                display: block;
                max-width: 250px;
                width: 100%;
            }
            .instructions {
                text-align: left;
                background: rgba(0,0,0,0.2);
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
            }
            .instructions h3 { margin: 0 0 15px 0; font-size: 16px; }
            .instructions ol { margin: 0; padding-left: 20px; }
            .instructions li { margin: 10px 0; font-size: 14px; line-height: 1.5; }
            .refresh-note { font-size: 12px; opacity: 0.7; margin-top: 15px; }
            .stats {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            .stat {
                background: rgba(255,255,255,0.1);
                padding: 10px 15px;
                border-radius: 10px;
                min-width: 80px;
            }
            .stat-value { font-size: 22px; font-weight: bold; }
            .stat-label { font-size: 10px; opacity: 0.8; text-transform: uppercase; }
            .error-box {
                background: rgba(255,0,0,0.2);
                border: 1px solid rgba(255,255,255,0.3);
                padding: 15px;
                border-radius: 10px;
                margin-top: 15px;
                text-align: left;
                font-size: 12px;
                word-break: break-word;
            }
            .trigger-badge {
                background: rgba(0,0,0,0.3);
                padding: 8px 15px;
                border-radius: 8px;
                font-size: 14px;
                margin: 10px 0;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 WhatsApp Patient Bot</h1>
            <p class="subtitle">Medical Image Clinical Profile Generator</p>
            <div class="model-badge">🤖 ${CONFIG.GEMINI_MODEL}</div>
    `;
    
    if (isConnected) {
        html += `
            <div class="status connected">✅ CONNECTED & RUNNING</div>
            <div class="trigger-badge">Send <strong>"."</strong> to process images</div>
            <div class="stats">
                <div class="stat"><div class="stat-value">${chatImageBuffers.size}</div><div class="stat-label">Active Chats</div></div>
                <div class="stat"><div class="stat-value">${totalBuffered}</div><div class="stat-label">Buffered Imgs</div></div>
                <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">Processed</div></div>
            </div>
            <div class="instructions">
                <h3>✨ How to use:</h3>
                <ol>
                    <li>Send medical report image(s) to the bot</li>
                    <li>Send more images if needed</li>
                    <li>Send <strong>.</strong> (period) when done</li>
                    <li>Bot generates Clinical Profile! 🎉</li>
                </ol>
            </div>
        `;
    } else if (qrCodeDataURL) {
        html += `
            <div class="status waiting">📲 SCAN QR CODE TO CONNECT</div>
            <div class="qr-container"><img src="${qrCodeDataURL}" alt="QR Code"></div>
            <div class="instructions">
                <h3>📋 To connect WhatsApp:</h3>
                <ol>
                    <li>Open <strong>WhatsApp</strong> on your phone</li>
                    <li>Tap <strong>⋮ Menu</strong> → <strong>Linked Devices</strong></li>
                    <li>Tap <strong>"Link a Device"</strong></li>
                    <li>Point camera at QR code above</li>
                </ol>
            </div>
            <p class="refresh-note">⟳ Page auto-refreshes every 5 seconds</p>
        `;
    } else if (lastError) {
        html += `
            <div class="status error">❌ ERROR</div>
            <p>Bot encountered an error</p>
            <div class="error-box">
                <strong>Status:</strong> ${botStatus}<br><br>
                <strong>Error:</strong> ${lastError}
            </div>
            <p class="refresh-note">⟳ Retrying automatically...</p>
        `;
    } else {
        html += `
            <div class="status waiting">⏳ ${botStatus.toUpperCase()}</div>
            <p>Please wait...</p>
            <p class="refresh-note">⟳ Page auto-refreshes every 5 seconds</p>
        `;
    }
    
    html += `</div></body></html>`;
    res.send(html);
});

app.get('/health', (req, res) => {
    res.json({ 
        status: 'running',
        connected: isConnected,
        botStatus: botStatus,
        lastError: lastError,
        model: CONFIG.GEMINI_MODEL,
        activeChats: chatImageBuffers.size,
        processedCount: processedCount,
        timestamp: new Date().toISOString()
    });
});

app.listen(PORT, () => {
    log('🌐', `Web server running on port ${PORT}`);
});

// ============== LOAD BAILEYS ==============
async function loadBaileys() {
    botStatus = 'Loading WhatsApp library...';
    log('📦', 'Loading Baileys library...');
    
    try {
        const baileys = await import('@whiskeysockets/baileys');
        
        makeWASocket = baileys.default || baileys.makeWASocket;
        useMultiFileAuthState = baileys.useMultiFileAuthState;
        DisconnectReason = baileys.DisconnectReason;
        downloadMediaMessage = baileys.downloadMediaMessage;
        fetchLatestBaileysVersion = baileys.fetchLatestBaileysVersion;
        
        log('✅', 'Baileys library loaded!');
        return true;
    } catch (error) {
        log('❌', `Failed to load Baileys: ${error.message}`);
        lastError = error.message;
        throw error;
    }
}

// ============== WHATSAPP BOT ==============
async function startBot() {
    try {
        botStatus = 'Initializing...';
        log('🚀', 'Starting WhatsApp Bot...');
        
        if (!makeWASocket) {
            await loadBaileys();
        }
        
        const authPath = join(__dirname, 'auth_session');
        
        botStatus = 'Loading auth state...';
        const { state, saveCreds } = await useMultiFileAuthState(authPath);
        
        botStatus = 'Fetching WhatsApp version...';
        let version;
        try {
            const versionData = await fetchLatestBaileysVersion();
            version = versionData.version;
            log('📱', `WhatsApp version: ${version.join('.')}`);
        } catch (e) {
            version = [2, 3000, 1015901307];
        }
        
        botStatus = 'Connecting...';
        log('🔌', 'Creating WhatsApp connection...');
        
        sock = makeWASocket({
            version,
            auth: state,
            logger: pino({ level: 'silent' }),
            browser: ['Ubuntu', 'Chrome', '20.0.04'],
            markOnlineOnConnect: false,
            getMessage: async () => ({ conversation: '' })
        });
        
        botStatus = 'Waiting for connection...';

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
                    lastError = null;
                    log('📱', '✅ QR Code ready - scan to connect!');
                } catch (err) {
                    lastError = `QR Error: ${err.message}`;
                }
            }
            
            if (connection === 'close') {
                isConnected = false;
                qrCodeDataURL = null;
                const statusCode = lastDisconnect?.error?.output?.statusCode;
                
                log('❌', `Disconnected: ${statusCode}`);
                
                if (statusCode === 401 || statusCode === 405 || statusCode === DisconnectReason?.loggedOut) {
                    try {
                        fs.rmSync(authPath, { recursive: true, force: true });
                        log('🗑️', 'Auth cleared');
                    } catch (e) {}
                }
                
                botStatus = 'Reconnecting...';
                setTimeout(startBot, 5000);
                
            } else if (connection === 'open') {
                isConnected = true;
                qrCodeDataURL = null;
                lastError = null;
                botStatus = 'Connected';
                log('✅', '🎉 CONNECTED TO WHATSAPP!');
                log('👀', 'Listening for messages...');
                log('💡', 'Send images, then send "." to process');
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
        lastError = error.message;
        botStatus = 'Error - retrying...';
        
        try {
            fs.rmSync(join(__dirname, 'auth_session'), { recursive: true, force: true });
        } catch (e) {}
        
        setTimeout(startBot, 10000);
    }
}

// ============== MESSAGE HANDLER ==============
async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    
    // Skip status broadcasts
    if (chatId === 'status@broadcast') return;
    
    const messageType = Object.keys(msg.message)[0];
    
    // Handle images
    if (messageType === 'imageMessage') {
        log('📷', `Image received from ${chatId.split('@')[0]}`);
        
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            
            // Initialize buffer for this chat if needed
            if (!chatImageBuffers.has(chatId)) {
                chatImageBuffers.set(chatId, []);
            }
            
            // Add image to this chat's buffer
            chatImageBuffers.get(chatId).push({
                data: buffer.toString('base64'),
                mimeType: msg.message.imageMessage.mimetype || 'image/jpeg',
                timestamp: Date.now()
            });
            
            const imageCount = chatImageBuffers.get(chatId).length;
            log('📦', `Chat buffer: ${imageCount} image(s)`);
            
            // Reset timeout for this chat
            if (chatTimeouts.has(chatId)) {
                clearTimeout(chatTimeouts.get(chatId));
            }
            
            // Set auto-clear timeout
            chatTimeouts.set(chatId, setTimeout(() => {
                if (chatImageBuffers.has(chatId)) {
                    log('⏰', `Auto-clearing ${chatImageBuffers.get(chatId).length} images for ${chatId.split('@')[0]}`);
                    chatImageBuffers.delete(chatId);
                    chatTimeouts.delete(chatId);
                }
            }, CONFIG.IMAGE_TIMEOUT_MS));
            
            // Send acknowledgment for first image
            if (imageCount === 1) {
                await sock.sendMessage(chatId, { 
                    text: `📷 Image received!\n\n_Send more images if needed, then send *.*  (period) to process._` 
                });
            } else {
                // Just react to subsequent images
                await sock.sendMessage(chatId, { 
                    react: { text: '📷', key: msg.key }
                });
            }
            
        } catch (error) {
            log('❌', `Download error: ${error.message}`);
        }
    }
    // Handle trigger text "."
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = (msg.message.conversation || msg.message.extendedTextMessage?.text || '').trim();
        
        // Check if it's the trigger
        if (text === CONFIG.TRIGGER_TEXT) {
            log('🔔', `Trigger received from ${chatId.split('@')[0]}`);
            
            // Check if there are images to process
            if (chatImageBuffers.has(chatId) && chatImageBuffers.get(chatId).length > 0) {
                // Clear timeout
                if (chatTimeouts.has(chatId)) {
                    clearTimeout(chatTimeouts.get(chatId));
                    chatTimeouts.delete(chatId);
                }
                
                // Get and clear images
                const images = chatImageBuffers.get(chatId);
                chatImageBuffers.delete(chatId);
                
                log('🔄', `Processing ${images.length} images...`);
                
                // Process images
                await processImages(sock, chatId, images);
            } else {
                // No images buffered
                await sock.sendMessage(chatId, { 
                    text: `❌ No images to process!\n\n_Send image(s) first, then send *.*  to generate the Clinical Profile._` 
                });
            }
        }
        // Help command
        else if (text.toLowerCase() === 'help' || text === '?') {
            await sock.sendMessage(chatId, { 
                text: `🏥 *WhatsApp Clinical Profile Bot*\n\n*How to use:*\n1️⃣ Send medical report image(s)\n2️⃣ Send more images if needed\n3️⃣ Send *.* (period) to process\n\n_The bot will analyze all images and generate a Clinical Profile._` 
            });
        }
        // Clear command
        else if (text.toLowerCase() === 'clear') {
            if (chatImageBuffers.has(chatId)) {
                const count = chatImageBuffers.get(chatId).length;
                chatImageBuffers.delete(chatId);
                if (chatTimeouts.has(chatId)) {
                    clearTimeout(chatTimeouts.get(chatId));
                    chatTimeouts.delete(chatId);
                }
                await sock.sendMessage(chatId, { 
                    text: `🗑️ Cleared ${count} buffered image(s).` 
                });
            } else {
                await sock.sendMessage(chatId, { 
                    text: `ℹ️ No images buffered.` 
                });
            }
        }
    }
}

// ============== GEMINI PROCESSING ==============
let model;
try {
    const genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);
    model = genAI.getGenerativeModel({ 
        model: CONFIG.GEMINI_MODEL,
        systemInstruction: CONFIG.SYSTEM_INSTRUCTION
    });
    log('✅', 'Gemini AI initialized');
} catch (error) {
    log('❌', `Gemini init error: ${error.message}`);
    lastError = `Gemini: ${error.message}`;
}

async function processImages(sock, chatId, images) {
    try {
        log('🤖', `Sending ${images.length} images to Gemini AI...`);
        
        // Send processing message
        await sock.sendMessage(chatId, { 
            text: `⏳ Processing ${images.length} image(s)...\n\n_Generating Clinical Profile, please wait..._` 
        });
        
        // Prepare image parts for Gemini
        const imageParts = images.map(img => ({
            inlineData: { data: img.data, mimeType: img.mimeType }
        }));
        
        // Call Gemini API
        const userPrompt = `Analyze these ${images.length} medical document image(s) and generate the Clinical Profile as per your instructions.`;
        
        const result = await model.generateContent([userPrompt, ...imageParts]);
        const response = await result.response;
        const clinicalProfile = response.text();
        
        log('✅', 'Clinical Profile generated!');
        processedCount++;
        
        // Log to console
        console.log('\n' + '═'.repeat(60));
        console.log(`📸 Images: ${images.length}`);
        console.log(`⏰ Time: ${new Date().toLocaleString()}`);
        console.log('═'.repeat(60));
        console.log(clinicalProfile);
        console.log('═'.repeat(60) + '\n');
        
        // Send to WhatsApp
        const maxLength = 4000;
        
        if (clinicalProfile.length <= maxLength) {
            await sock.sendMessage(chatId, { text: clinicalProfile });
        } else {
            // Split long messages
            await sock.sendMessage(chatId, { 
                text: clinicalProfile.substring(0, maxLength - 20) + '\n\n_(continued...)_'
            });
            
            let remaining = clinicalProfile.substring(maxLength - 20);
            while (remaining.length > 0) {
                await new Promise(r => setTimeout(r, 1000));
                const chunk = remaining.substring(0, maxLength);
                remaining = remaining.substring(maxLength);
                await sock.sendMessage(chatId, { text: chunk });
            }
        }
        
        log('📤', 'Clinical Profile sent!');
        
    } catch (error) {
        log('❌', `Gemini Error: ${error.message}`);
        console.error(error);
        
        await sock.sendMessage(chatId, { 
            text: `❌ Error generating Clinical Profile\n\n_${error.message}_\n\nPlease try again.` 
        });
    }
}

// ============== START ==============
console.log('\n╔════════════════════════════════════════════════════════╗');
console.log('║     WhatsApp Patient Bot - Clinical Profile Generator  ║');
console.log('╚════════════════════════════════════════════════════════╝\n');

log('🏁', 'Initializing...');
log('🤖', `Model: ${CONFIG.GEMINI_MODEL}`);
log('🔔', `Trigger: Send "${CONFIG.TRIGGER_TEXT}" to process images`);

startBot();
