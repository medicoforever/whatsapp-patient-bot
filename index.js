import makeWASocket, { 
    useMultiFileAuthState, 
    DisconnectReason,
    downloadMediaMessage 
} from '@whiskeysockets/baileys';
import { GoogleGenerativeAI } from '@google/generative-ai';
import pino from 'pino';
import QRCode from 'qrcode';
import express from 'express';
import fs from 'fs';

// ============== CONFIGURATION ==============
const CONFIG = {
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    
    // Using the newest Gemini model
    GEMINI_MODEL: 'gemini-3-flash-preview',
    
    // Time to wait for patient name after images (2 minutes)
    IMAGE_TIMEOUT_MS: 120000,
    
    // Send results back to WhatsApp
    SEND_TO_WHATSAPP: true,
    
    // Your exact system instruction
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
let imageBuffer = [];
let bufferTimeout = null;
let sock = null;
let isConnected = false;
let qrCodeDataURL = null;
let processedCount = 0;

// Gemini setup with system instruction
const genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ 
    model: CONFIG.GEMINI_MODEL,
    systemInstruction: CONFIG.SYSTEM_INSTRUCTION
});

function log(emoji, message) {
    const time = new Date().toLocaleTimeString();
    console.log(`[${time}] ${emoji} ${message}`);
}

// ============== WEB SERVER FOR QR CODE ==============
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
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
            h1 { 
                margin: 0 0 10px 0; 
                font-size: 24px;
            }
            .subtitle {
                opacity: 0.9;
                margin-bottom: 20px;
                font-size: 14px;
            }
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
            .connected { 
                background: #4CAF50; 
                animation: pulse 2s infinite;
            }
            .waiting { 
                background: rgba(255,255,255,0.2); 
            }
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
            .instructions h3 {
                margin: 0 0 15px 0;
                font-size: 16px;
            }
            .instructions ol {
                margin: 0;
                padding-left: 20px;
            }
            .instructions li {
                margin: 10px 0;
                font-size: 14px;
                line-height: 1.5;
            }
            .refresh-note {
                font-size: 12px;
                opacity: 0.7;
                margin-top: 15px;
            }
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
            .stat-value {
                font-size: 22px;
                font-weight: bold;
            }
            .stat-label {
                font-size: 10px;
                opacity: 0.8;
                text-transform: uppercase;
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
            <div class="status connected">
                ✅ CONNECTED & RUNNING
            </div>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">24/7</div>
                    <div class="stat-label">Uptime</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${imageBuffer.length}</div>
                    <div class="stat-label">Buffered</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${processedCount}</div>
                    <div class="stat-label">Processed</div>
                </div>
            </div>
            <div class="instructions">
                <h3>✨ How to use:</h3>
                <ol>
                    <li>Send patient's medical report images to any WhatsApp group</li>
                    <li>Then send patient identifier as text (e.g., "Patient 1" or any name)</li>
                    <li>Bot will generate Clinical Profile and reply!</li>
                </ol>
            </div>
        `;
    } else if (qrCodeDataURL) {
        html += `
            <div class="status waiting">
                📲 SCAN QR CODE TO CONNECT
            </div>
            <div class="qr-container">
                <img src="${qrCodeDataURL}" alt="QR Code">
            </div>
            <div class="instructions">
                <h3>📋 To connect WhatsApp:</h3>
                <ol>
                    <li>Open <strong>WhatsApp</strong> on your phone</li>
                    <li>Tap <strong>⋮ Menu</strong> (3 dots) → <strong>Linked Devices</strong></li>
                    <li>Tap <strong>"Link a Device"</strong></li>
                    <li>Point camera at QR code above</li>
                </ol>
            </div>
            <p class="refresh-note">⟳ Page auto-refreshes every 5 seconds</p>
        `;
    } else {
        html += `
            <div class="status waiting">
                ⏳ INITIALIZING...
            </div>
            <p>Please wait, generating QR code...</p>
            <p class="refresh-note">⟳ Page auto-refreshes every 5 seconds</p>
        `;
    }
    
    html += `
        </div>
    </body>
    </html>
    `;
    
    res.send(html);
});

app.get('/health', (req, res) => {
    res.json({ 
        status: 'running',
        connected: isConnected,
        model: CONFIG.GEMINI_MODEL,
        bufferedImages: imageBuffer.length,
        processedCount: processedCount,
        timestamp: new Date().toISOString()
    });
});

app.listen(PORT, () => {
    log('🌐', `Web server running on port ${PORT}`);
    log('🤖', `Using Gemini model: ${CONFIG.GEMINI_MODEL}`);
});

// ============== WHATSAPP BOT ==============
async function startBot() {
    log('🚀', 'Starting WhatsApp Bot...');
    
    const { state, saveCreds } = await useMultiFileAuthState('auth_session');
    
    sock = makeWASocket.default({
        auth: state,
        printQRInTerminal: true,
        logger: pino({ level: 'silent' }),
        browser: ['Patient Bot', 'Chrome', '120.0.0']
    });

    sock.ev.on('connection.update', async (update) => {
        const { connection, lastDisconnect, qr } = update;
        
        if (qr) {
            try {
                qrCodeDataURL = await QRCode.toDataURL(qr, {
                    width: 300,
                    margin: 2,
                    color: {
                        dark: '#128C7E',
                        light: '#FFFFFF'
                    }
                });
                isConnected = false;
                log('📱', 'QR Code generated - open web page to scan');
            } catch (err) {
                log('❌', `QR generation error: ${err.message}`);
            }
        }
        
        if (connection === 'close') {
            isConnected = false;
            qrCodeDataURL = null;
            const statusCode = lastDisconnect?.error?.output?.statusCode;
            const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
            
            log('❌', `Connection closed. Code: ${statusCode}`);
            
            if (shouldReconnect) {
                log('🔄', 'Reconnecting in 5 seconds...');
                setTimeout(startBot, 5000);
            } else {
                log('🚪', 'Logged out. Clearing session...');
                try {
                    fs.rmSync('auth_session', { recursive: true, force: true });
                } catch (e) {}
                setTimeout(startBot, 5000);
            }
        } else if (connection === 'open') {
            isConnected = true;
            qrCodeDataURL = null;
            log('✅', 'CONNECTED TO WHATSAPP!');
            log('👀', 'Listening for messages in groups...');
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
}

// ============== MESSAGE HANDLER ==============
async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    
    // Only process group messages
    if (!chatId.endsWith('@g.us')) return;
    
    const messageType = Object.keys(msg.message)[0];
    
    // Handle images
    if (messageType === 'imageMessage') {
        log('📷', `Image received`);
        
        try {
            const buffer = await downloadMediaMessage(msg, 'buffer', {});
            
            imageBuffer.push({
                data: buffer.toString('base64'),
                mimeType: msg.message.imageMessage.mimetype || 'image/jpeg',
                timestamp: Date.now(),
                groupId: chatId
            });
            
            log('📦', `Buffer: ${imageBuffer.length} image(s)`);
            
            if (bufferTimeout) clearTimeout(bufferTimeout);
            
            bufferTimeout = setTimeout(() => {
                if (imageBuffer.length > 0) {
                    log('⏰', 'Timeout - clearing buffer');
                    imageBuffer = [];
                }
            }, CONFIG.IMAGE_TIMEOUT_MS);
            
        } catch (error) {
            log('❌', `Download error: ${error.message}`);
        }
    }
    // Handle text (patient identifier)
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = msg.message.conversation || 
                     msg.message.extendedTextMessage?.text || '';
        
        if (text && text.trim() && imageBuffer.length > 0) {
            const patientIdentifier = text.trim();
            const groupImages = imageBuffer.filter(img => img.groupId === chatId);
            
            if (groupImages.length > 0) {
                log('👤', `Patient ID: "${patientIdentifier}"`);
                log('🔄', `Processing ${groupImages.length} images...`);
                
                if (bufferTimeout) {
                    clearTimeout(bufferTimeout);
                    bufferTimeout = null;
                }
                
                const imagesToProcess = [...groupImages];
                imageBuffer = imageBuffer.filter(img => img.groupId !== chatId);
                
                await processPatientImages(sock, chatId, patientIdentifier, imagesToProcess);
            }
        }
    }
}

// ============== GEMINI PROCESSING ==============
async function processPatientImages(sock, chatId, patientIdentifier, images) {
    try {
        log('🤖', `Sending ${images.length} images to Gemini AI...`);
        
        // Send processing message
        await sock.sendMessage(chatId, { 
            text: `⏳ Processing ${images.length} image(s) for *${patientIdentifier}*...\n\n_Generating Clinical Profile..._` 
        });
        
        // Prepare image parts for Gemini
        const imageParts = images.map(img => ({
            inlineData: {
                data: img.data,
                mimeType: img.mimeType
            }
        }));
        
        // Simple prompt - system instruction handles the rest
        const userPrompt = `Analyze these ${images.length} medical document image(s) and generate the Clinical Profile as per your instructions.`;
        
        // Call Gemini API
        const result = await model.generateContent([userPrompt, ...imageParts]);
        const response = await result.response;
        const clinicalProfile = response.text();
        
        log('✅', `Clinical Profile generated!`);
        processedCount++;
        
        // Log to console
        console.log('\n' + '═'.repeat(60));
        console.log(`📋 Patient: ${patientIdentifier}`);
        console.log(`📸 Images: ${images.length}`);
        console.log(`⏰ Time: ${new Date().toLocaleString()}`);
        console.log('═'.repeat(60));
        console.log(clinicalProfile);
        console.log('═'.repeat(60) + '\n');
        
        // Send to WhatsApp
        if (CONFIG.SEND_TO_WHATSAPP) {
            // Create header
            const header = `📋 *Report for: ${patientIdentifier}*\n📸 _${images.length} image(s) analyzed_\n${'─'.repeat(30)}\n\n`;
            
            const maxLength = 4000;
            
            if (clinicalProfile.length + header.length <= maxLength) {
                await sock.sendMessage(chatId, { text: header + clinicalProfile });
            } else {
                // Split long messages
                await sock.sendMessage(chatId, { 
                    text: header + clinicalProfile.substring(0, maxLength - header.length - 20) + '\n\n_(continued...)_'
                });
                
                let remaining = clinicalProfile.substring(maxLength - header.length - 20);
                while (remaining.length > 0) {
                    await new Promise(r => setTimeout(r, 1000));
                    const chunk = remaining.substring(0, maxLength);
                    remaining = remaining.substring(maxLength);
                    await sock.sendMessage(chatId, { text: chunk });
                }
            }
            
            log('📤', 'Clinical Profile sent to WhatsApp!');
        }
        
    } catch (error) {
        log('❌', `Gemini Error: ${error.message}`);
        console.error(error);
        
        // Send error message
        await sock.sendMessage(chatId, { 
            text: `❌ Error processing *${patientIdentifier}*\n\n_${error.message}_\n\nPlease try again.` 
        });
    }
}

// ============== START ==============
console.log('');
console.log('╔════════════════════════════════════════════════════════╗');
console.log('║     WhatsApp Patient Bot - Clinical Profile Generator  ║');
console.log('╚════════════════════════════════════════════════════════╝');
console.log('');

log('🏁', 'Initializing...');
log('🤖', `Model: ${CONFIG.GEMINI_MODEL}`);

startBot().catch(err => {
    log('💥', `Fatal: ${err.message}`);
    console.error(err);
    setTimeout(startBot, 10000);
});
