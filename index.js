const { 
    makeWASocket, 
    useMultiFileAuthState, 
    DisconnectReason,
    downloadMediaMessage 
} = require('@whiskeysockets/baileys');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const pino = require('pino');
const qrcode = require('qrcode-terminal');
const express = require('express');
const fs = require('fs');

// ============== CONFIGURATION ==============
const CONFIG = {
    GEMINI_API_KEY: process.env.GEMINI_API_KEY,
    IMAGE_TIMEOUT_MS: 120000,
    SEND_TO_WHATSAPP: true,
    ANALYSIS_PROMPT: `You are a medical image analysis assistant.

Analyze these medical images and provide:
1. Key observations from each image
2. Overall summary of findings
3. Any notable patterns or concerns

Be thorough but concise. Format your response clearly with sections.`
};

// ============== SETUP ==============
let imageBuffer = [];
let bufferTimeout = null;
let sock = null;
let isConnected = false;
let qrCodeData = null;

const genAI = new GoogleGenerativeAI(CONFIG.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

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
        <meta http-equiv="refresh" content="10">
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                box-sizing: border-box;
            }
            .container {
                text-align: center;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                max-width: 400px;
                width: 100%;
            }
            h1 { margin-bottom: 10px; }
            .status {
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                font-size: 18px;
            }
            .connected { background: #4CAF50; }
            .waiting { background: #FF9800; }
            .qr-container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                display: inline-block;
                margin: 20px 0;
            }
            .qr-code {
                font-family: monospace;
                font-size: 4px;
                line-height: 4px;
                color: black;
                white-space: pre;
            }
            .instructions {
                text-align: left;
                background: rgba(0,0,0,0.2);
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .instructions ol {
                margin: 10px 0;
                padding-left: 20px;
            }
            .instructions li {
                margin: 8px 0;
            }
            .refresh-note {
                font-size: 12px;
                opacity: 0.8;
                margin-top: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 WhatsApp Patient Bot</h1>
    `;
    
    if (isConnected) {
        html += `
            <div class="status connected">
                ✅ CONNECTED & RUNNING
            </div>
            <p>The bot is actively monitoring your WhatsApp groups.</p>
            <div class="instructions">
                <strong>How to use:</strong>
                <ol>
                    <li>Send images to any WhatsApp group</li>
                    <li>Then send patient name as text</li>
                    <li>Bot will analyze and reply!</li>
                </ol>
            </div>
        `;
    } else if (qrCodeData) {
        html += `
            <div class="status waiting">
                ⏳ WAITING FOR QR SCAN
            </div>
            <div class="qr-container">
                <div class="qr-code">${qrCodeData}</div>
            </div>
            <div class="instructions">
                <strong>To connect:</strong>
                <ol>
                    <li>Open WhatsApp on your phone</li>
                    <li>Tap ⋮ Menu → Linked Devices</li>
                    <li>Tap "Link a Device"</li>
                    <li>Scan the QR code above</li>
                </ol>
            </div>
            <p class="refresh-note">Page refreshes every 10 seconds</p>
        `;
    } else {
        html += `
            <div class="status waiting">
                ⏳ STARTING...
            </div>
            <p>Please wait, QR code loading...</p>
            <p class="refresh-note">Page refreshes every 10 seconds</p>
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
        timestamp: new Date().toISOString()
    });
});

app.listen(PORT, () => {
    log('🌐', `Web server running on port ${PORT}`);
});

// ============== WHATSAPP BOT ==============
async function startBot() {
    log('🚀', 'Starting WhatsApp Bot...');
    
    const { state, saveCreds } = await useMultiFileAuthState('auth_session');
    
    sock = makeWASocket({
        auth: state,
        printQRInTerminal: true,
        logger: pino({ level: 'silent' }),
        browser: ['Patient Bot', 'Chrome', '120.0.0']
    });

    sock.ev.on('connection.update', async (update) => {
        const { connection, lastDisconnect, qr } = update;
        
        if (qr) {
            qrCodeData = generateTextQR(qr);
            isConnected = false;
            log('📱', 'QR Code generated - check web page or console');
            console.log('\n--- SCAN THIS QR CODE ---\n');
            qrcode.generate(qr, { small: true });
            console.log('\n--- OR OPEN WEB PAGE ---\n');
        }
        
        if (connection === 'close') {
            isConnected = false;
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
            qrCodeData = null;
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

function generateTextQR(qrString) {
    const QRCode = require('qrcode-terminal');
    let qrText = '';
    
    const originalLog = console.log;
    console.log = (text) => { qrText += text + '\n'; };
    
    QRCode.generate(qrString, { small: true }, (qr) => {
        qrText = qr;
    });
    
    console.log = originalLog;
    
    return qrText.replace(/\n/g, '<br>').replace(/ /g, '&nbsp;');
}

async function handleMessage(sock, msg) {
    const chatId = msg.key.remoteJid;
    
    if (!chatId.endsWith('@g.us')) return;
    
    const messageType = Object.keys(msg.message)[0];
    
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
    else if (messageType === 'conversation' || messageType === 'extendedTextMessage') {
        const text = msg.message.conversation || 
                     msg.message.extendedTextMessage?.text || '';
        
        if (text && text.trim() && imageBuffer.length > 0) {
            const patientName = text.trim();
            const groupImages = imageBuffer.filter(img => img.groupId === chatId);
            
            if (groupImages.length > 0) {
                log('👤', `Patient: "${patientName}"`);
                log('🔄', `Processing ${groupImages.length} images...`);
                
                if (bufferTimeout) {
                    clearTimeout(bufferTimeout);
                    bufferTimeout = null;
                }
                
                const imagesToProcess = [...groupImages];
                imageBuffer = imageBuffer.filter(img => img.groupId !== chatId);
                
                await processPatientImages(sock, chatId, patientName, imagesToProcess);
            }
        }
    }
}

async function processPatientImages(sock, chatId, patientName, images) {
    try {
        log('🤖', `Sending to Gemini AI...`);
        
        await sock.sendMessage(chatId, { 
            text: `⏳ Analyzing ${images.length} image(s) for *${patientName}*...\n\nPlease wait...` 
        });
        
        const imageParts = images.map(img => ({
            inlineData: {
                data: img.data,
                mimeType: img.mimeType
            }
        }));
        
        const fullPrompt = `
Patient Name: ${patientName}
Number of Images: ${images.length}
Date/Time: ${new Date().toLocaleString()}

${CONFIG.ANALYSIS_PROMPT}
        `.trim();
        
        const result = await model.generateContent([fullPrompt, ...imageParts]);
        const response = await result.response;
        const analysisText = response.text();
        
        log('✅', `Analysis complete!`);
        
        console.log('\n' + '═'.repeat(50));
        console.log(`📋 ${patientName}`);
        console.log('═'.repeat(50));
        console.log(analysisText);
        console.log('═'.repeat(50) + '\n');
        
        if (CONFIG.SEND_TO_WHATSAPP) {
            const maxLength = 4000;
            const header = `📋 *Analysis: ${patientName}*\n📸 Images: ${images.length}\n${'─'.repeat(25)}\n\n`;
            
            if (analysisText.length + header.length <= maxLength) {
                await sock.sendMessage(chatId, { text: header + analysisText });
            } else {
                await sock.sendMessage(chatId, { 
                    text: header + analysisText.substring(0, maxLength - header.length - 20) + '\n\n_(continued...)_'
                });
                
                let remaining = analysisText.substring(maxLength - header.length - 20);
                while (remaining.length > 0) {
                    await new Promise(r => setTimeout(r, 1000));
                    const chunk = remaining.substring(0, maxLength);
                    remaining = remaining.substring(maxLength);
                    await sock.sendMessage(chatId, { text: chunk });
                }
            }
            
            log('📤', 'Sent to WhatsApp!');
        }
        
    } catch (error) {
        log('❌', `Error: ${error.message}`);
        
        await sock.sendMessage(chatId, { 
            text: `❌ Error analyzing *${patientName}*\n\n${error.message}` 
        });
    }
}

// ============== START ==============
log('🏁', 'Initializing...');
startBot().catch(err => {
    log('💥', `Fatal: ${err.message}`);
    console.error(err);
    setTimeout(startBot, 10000);
});
