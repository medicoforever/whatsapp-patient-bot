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
import crypto from 'crypto';

// Setup FFmpeg path automatically for Render
ffmpeg.setFfmpegPath(ffmpegInstaller.path);

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Helper to parse keys from comma-separated string
const getApiKeys = () => {
  const keys = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
  return keys.split(',').map(k => k.trim()).filter(k => k.length > 0);
};

// ======================================================================
// 🟢 NEW CONFIGURATION AREA
// ======================================================================

const SECONDARY_SYSTEM_INSTRUCTION = `You are an expert radiologist. When you receive a context, it is mostly about a patient and sometimes they might have been advised with any imaging modality. You analyse that info and then advise regarding that as an expert radiologist what to be seen in that specific imaging modality for that specific patient including various hypothetical imaging findings from common to less common for that patient condition in that specific imaging modality. suppose of you cant indentify thr specific imaging modality in thr given context, you yourself choose the appropriate imaging modality based on the specific conditions context`;

const SECONDARY_TRIGGER_PROMPT = `Here is the Clinical Profile generated from the patient's reports. Please analyze this profile according to your system instructions and provide the final output.`;

// ======================================================================

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

  // 🔗 Retention & expiry timeouts: 12 hours
  CONTEXT_RETENTION_MS: 12 * 60 * 60 * 1000, // 12 hours
  MAX_STORED_CONTEXTS: 1000, // 🔧 Increased from 20 to 1000 to prevent premature eviction
  COMMANDS:['.', '.1', '.2', '.3', '..', '..1', '..2', '..3', 'help', '?', 'clear', 'status'],
  TYPING_DELAY_MIN: 3000,
  TYPING_DELAY_MAX: 6000,
  SUPPORTED_AUDIO_MIMES:[
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 'audio/x-wav',
    'audio/ogg', 'audio/opus', 'audio/aac', 'audio/m4a', 'audio/x-m4a',
    'audio/mp4', 'audio/flac', 'audio/webm', 'audio/amr', 'audio/3gpp'
  ],
  SUPPORTED_AUDIO_EXTENSIONS:['.mp3', '.wav', '.ogg', '.opus', '.m4a', '.aac', '.flac', '.webm', '.amr', '.3gp'],
  SUPPORTED_VIDEO_MIMES:[
    'video/mp4', 'video/mpeg', 'video/webm', 'video/x-msvideo', 'video/avi',
    'video/quicktime', 'video/x-matroska', 'video/mkv', 'video/3gpp', 'video/3gp'
  ],
  SUPPORTED_VIDEO_EXTENSIONS:['.mp4', '.mpeg', '.mpg', '.webm', '.avi', '.mov', '.mkv', '.3gp'],
  // 🔗 Media Viewer URL expiry: 12 hours
  MEDIA_VIEWER_EXPIRY_MS: 12 * 60 * 60 * 1000,
  // 🔧 Decryption failure auto-heal settings
  DECRYPT_FAIL_THRESHOLD: 8,
  DECRYPT_FAIL_WINDOW_MS: 60000,
  // 🔄 Retry settings for empty source group messages
  EMPTY_MSG_RETRY_DELAY_MS: 5000, // Wait 5 seconds before retry
  EMPTY_MSG_MAX_RETRIES: 10, // Max retries for empty messages (50s total buffer)
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

IMPORTANT ADDITIONAL OUTPUT:
After the Clinical Profile paragraph, you MUST output a second line (separated by a blank line) in EXACTLY this format:
<<JSON>>{"age":"<age or unknown>","sex":"<M/F/unknown>","study":"<imaging study indicated or Not mentioned>","brief":"<very concise reason for scan using abbreviations like H/o, c/o, s/o, etc., mentioning duration of symptoms>"}<<JSON>>

Rules for the JSON line:
- age: Extract patient age from the content. If not found, use "unknown".
- sex: Extract patient sex/gender from the content. Use "M" for male, "F" for female. If not found, use "unknown".
- study: The imaging study that is currently indicated/requested (e.g., "CT Thorax", "MRI Brain", "USG Abdomen"). If not obvious from the content, use "Not mentioned".
- brief: A very short clinical summary (max 15-20 words) using medical abbreviations. Example: "H/o fever 4d, cough, SOB, k/c ILD - r/o infective exacerbation" or "Giddiness 15d, slurred speech, R UL weakness, k/c HTN/DM - r/o cerebellar infarct"

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
   - Also include the updated <<JSON>> line

3. If the user sends ADDITIONAL FILES in the reply:
   - Analyze the new files along with the original context
   - Generate an UPDATED Clinical Profile that includes information from all files
   - Also include the updated <<JSON>> line

IMPORTANT: Always identify whether the user is asking a question or providing additional information, and respond appropriately.`
};

// ======================================================================
// 🔗 MEDIA VIEWER STORE (In-Memory with 12hr Expiry)
// ======================================================================
const mediaViewerStore = new Map();

function storeMediaForViewer(mediaFiles) {
  const viewerId = crypto.randomBytes(16).toString('hex');
  const expiresAt = Date.now() + CONFIG.MEDIA_VIEWER_EXPIRY_MS;

  const viewableMedia =[];
  for (const m of mediaFiles) {
    if (m.type === 'image' || m.type === 'pdf' || m.type === 'audio' || m.type === 'voice' || m.type === 'video') {
      viewableMedia.push({
        type: m.type,
        data: m.data,
        mimeType: m.mimeType,
        caption: m.caption || '',
        fileName: m.fileName || ''
      });
    }
  }

  if (viewableMedia.length === 0) return null;

  mediaViewerStore.set(viewerId, {
    media: viewableMedia,
    expiresAt: expiresAt,
    createdAt: Date.now()
  });

  log('🔗', `Media viewer created: ${viewerId} (${viewableMedia.length} files, expires in 12h)`);

  setTimeout(() => {
    if (mediaViewerStore.has(viewerId)) {
      mediaViewerStore.delete(viewerId);
      log('🧹', `Media viewer expired and cleaned: ${viewerId}`);
    }
  }, CONFIG.MEDIA_VIEWER_EXPIRY_MS);

  return viewerId;
}

setInterval(() => {
  const now = Date.now();
  let cleaned = 0;
  for (const[id, entry] of mediaViewerStore) {
    if (now >= entry.expiresAt) {
      mediaViewerStore.delete(id);
      cleaned++;
    }
  }
  if (cleaned > 0) {
    log('🧹', `Periodic cleanup: removed ${cleaned} expired media viewer(s)`);
  }
}, 60 * 60 * 1000);

// ======================================================================
// 🔧 DECRYPTION FAILURE TRACKER (Auto-Heal)
// ======================================================================
let decryptFailTimestamps =[];
let isHealingInProgress = false;
let startupHealDone = false;

function trackDecryptionFailure() {
  const now = Date.now();
  decryptFailTimestamps.push(now);
  // Keep only timestamps within the window
  decryptFailTimestamps = decryptFailTimestamps.filter(
    ts => (now - ts) < CONFIG.DECRYPT_FAIL_WINDOW_MS
  );

  if (decryptFailTimestamps.length >= CONFIG.DECRYPT_FAIL_THRESHOLD && !isHealingInProgress) {
    log('🚨', `Decryption failure threshold reached (${decryptFailTimestamps.length} failures in ${CONFIG.DECRYPT_FAIL_WINDOW_MS / 1000}s). Triggering auto-heal...`);
    triggerSessionHeal();
  }
}

async function nukeSessionKeysFromMongo() {
  if (!mongoConnected || !SessionModel) {
    log('⚠️', ' MongoDB not available for key cleanup');
    return 0;
  }
  try {
    const result = await SessionModel.deleteMany({
      key: { $regex: /^key_/ }
    });
    log('🗑', ` Nuked ${result.deletedCount} signal session keys from MongoDB`);
    return result.deletedCount;
  } catch (error) {
    log('❌', ` Failed to nuke keys: ${error.message}`);
    return 0;
  }
}

async function triggerSessionHeal(reason = 'threshold') {
  if (isHealingInProgress) return;
  isHealingInProgress = true;

  log('🔧', '══════════════════════════════════════');
  log('🔧', ' AUTO-HEAL: Signal session key reset ');
  log('🔧', '══════════════════════════════════════');

  try {
    const deleted = await nukeSessionKeysFromMongo();
    decryptFailTimestamps =[];

    if (deleted > 0) {
      log('🔧', ` Cleared ${deleted} corrupted keys. Auth creds preserved ✅`);
    }

    if (sock) {
      log('🔄', ' Forcing reconnection...');
      try {
        sock.end(new Error(`Session heal: ${reason}`));
      } catch (e) {
        log('⚠️', ` Socket close (harmless): ${e.message}`);
      }
    }

    setTimeout(() => {
      isHealingInProgress = false;
      log('🔧', ' Heal cooldown complete.');
    }, 30000);

  } catch (error) {
    log('❌', ` Heal error: ${error.message}`);
    isHealingInProgress = false;
  }
}

// ======================================================================

// ======================================================================
// 🔄 API KEY ROTATION LOGIC (Every 2 Hours)
// ======================================================================
function rotateApiKeys() {
  if (CONFIG.API_KEYS.length > 1) {
    const key = CONFIG.API_KEYS.shift();
    CONFIG.API_KEYS.push(key);
    log('🔄', `API Keys Rotated. New primary key starts with: ...${CONFIG.API_KEYS[0].slice(-4)}`);
  }
}

setInterval(rotateApiKeys, 2 * 60 * 60 * 1000);
// ======================================================================

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

  const questionStarters =[
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

  const questionPhrases =[
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
      log('🗑', 'Cleared all MongoDB sessions');
    } catch (error) {
      log('❌', `MongoDB clear error: ${error.message}`);
    }
  };

  const clearSessionKeys = async () => {
    try {
      const result = await SessionModel.deleteMany({
        key: { $regex: /^key_/ }
      });
      log('🗑', `Cleared ${result.deletedCount} signal session keys from MongoDB`);
    } catch (error) {
      log('❌', `MongoDB session key clear error: ${error.message}`);
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
        // 🚀 MASSIVE SPEED BOOST HERE: Using BulkWrite to prevent WhatsApp timeouts
        set: async (data) => {
          const bulkOps = [];
          for (const[type, entries] of Object.entries(data)) {
            for (const [id, value] of Object.entries(entries)) {
              const key = `key_${type}_${id}`;
              if (value) {
                const serialized = JSON.stringify(value, (k, v) => {
                  if (typeof v === 'bigint') return { type: 'BigInt', value: v.toString() };
                  if (v instanceof Uint8Array) return { type: 'Uint8Array', value: Array.from(v) };
                  if (Buffer.isBuffer(v)) return { type: 'Buffer', value: Array.from(v) };
                  return v;
                });
                bulkOps.push({
                  updateOne: {
                    filter: { key },
                    update: { $set: { key, value: serialized, updatedAt: new Date() } },
                    upsert: true
                  }
                });
              } else {
                bulkOps.push({ deleteOne: { filter: { key } } });
              }
            }
          }
          
          if (bulkOps.length > 0) {
            try {
              // ordered: false tells MongoDB to do them all simultaneously
              await SessionModel.bulkWrite(bulkOps, { ordered: false });
            } catch (error) {
              log('❌', `MongoDB bulk write error: ${error.message}`);
            }
          }
        }
      }
    },
    saveCreds: async () => {
      await writeData('auth_creds', creds);
      log('💾', 'Credentials saved to MongoDB');
    },
    clearAll,
    clearSessionKeys
  };
}

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
      log('🗑', 'Cleared all MongoDB sessions');
    } catch (error) {
      log('❌', `MongoDB clear error: ${error.message}`);
    }
  };

  // 🔧 NEW: Clear only signal session keys, keep auth creds
  const clearSessionKeys = async () => {
    try {
      const result = await SessionModel.deleteMany({
        key: { $regex: /^key_/ }
      });
      log('🗑', `Cleared ${result.deletedCount} signal session keys from MongoDB`);
    } catch (error) {
      log('❌', `MongoDB session key clear error: ${error.message}`);
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

const chatMediaBuffers = new Map();
const chatTimeouts = new Map();
const chatContexts = new Map();
const botMessageIds = new Map();

// ======================================================================
// 🆔 MESSAGE DEDUPLICATION (prevents duplicate processing after reconnects)
// ======================================================================
const processedMessageIds = new Set();
const MAX_PROCESSED_IDS = 5000;

function isMessageAlreadyProcessed(msgId) {
  if (!msgId) return false;
  if (processedMessageIds.has(msgId)) {
    return true;
  }
  processedMessageIds.add(msgId);
  // Prevent unbounded growth
  if (processedMessageIds.size > MAX_PROCESSED_IDS) {
    const arr = Array.from(processedMessageIds);
    arr.slice(0, Math.floor(MAX_PROCESSED_IDS / 2)).forEach(id => processedMessageIds.delete(id));
  }
  return false;
}
            
// ======================================================================

// ======================================================================
// 🔄 PENDING EMPTY MESSAGES TRACKER (for source group retry)
// ======================================================================
const pendingEmptyMessages = new Map(); // msgId -> { msg, retryCount, chatId, timestamp }
// ======================================================================

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
    chatBuffer.set(senderId,[]);
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
  return[];
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

// ======================================================================
// 🧠 HELPER: Smart Grouping by Caption
// ======================================================================
function groupMediaSmartly(mediaFiles) {
  const distinctCaptions = new Set();
  mediaFiles.forEach(f => {
    if (f.caption && f.caption.trim()) {
      distinctCaptions.add(f.caption.trim());
    }
  });

  if (distinctCaptions.size <= 1) {
    return[mediaFiles];
  }

  const batches =[];
  let currentBatch =[];
  let activeCaption = null;

  for (const file of mediaFiles) {
    const fileCaption = (file.caption || '').trim();

    if (fileCaption && fileCaption !== activeCaption) {
      if (currentBatch.length > 0) {
        batches.push(currentBatch);
      }
      currentBatch =[file];
      activeCaption = fileCaption;
    } else {
      currentBatch.push(file);

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

// ======================================================================
// 🔄 UPDATED TIMEOUT LOGIC (Includes Smart Batching)
// ======================================================================
function resetUserTimeout(chatId, senderId, senderName) {
  if (!chatTimeouts.has(chatId)) {
    chatTimeouts.set(chatId, new Map());
  }
  const chatTimeoutMap = chatTimeouts.get(chatId);

  if (chatTimeoutMap.has(senderId)) {
    clearTimeout(chatTimeoutMap.get(senderId));
  }

  const isCTSource = chatId === CONFIG.GROUPS.CT_SOURCE;
  const isMRISource = chatId === CONFIG.GROUPS.MRI_SOURCE;
  const isAutoGroup = isCTSource || isMRISource;

  const delay = isAutoGroup ? CONFIG.AUTO_PROCESS_DELAY_MS : CONFIG.MEDIA_TIMEOUT_MS;

  const shortId = getShortSenderId(senderId);

  const timeoutCallback = async () => {
    try {
      if (isAutoGroup) {
        const mediaFiles = clearUserBuffer(chatId, senderId);
        if (mediaFiles.length > 0) {
          log('⏱️', `Auto-processing ${mediaFiles.length} item(s) from Source Group (${isCTSource ? 'CT' : 'MRI'})`);

          const targetChatId = isCTSource ? CONFIG.GROUPS.CT_TARGET : CONFIG.GROUPS.MRI_TARGET;

          if (targetChatId) {
            const batches = groupMediaSmartly(mediaFiles);

            if (batches.length > 1) {
              log('🔀', `Detected ${batches.length} distinct patient contexts. Processing separately.`);
            }

            for (let i = 0; i < batches.length; i++) {
              const batch = batches[i];
              if (batch.length === 0) continue;

              log('▶️', `Processing Batch ${i+1}/${batches.length} (${batch.length} files)`);

              await processMedia(chatId, batch, false, null, senderId, senderName, null, 3, false, targetChatId);

              if (i < batches.length - 1) {
                await new Promise(r => setTimeout(r, 2000));
              }
            }

          } else {
            log('⚠️', 'Target group not configured for this source!');
          }
        }
      } else {
        const clearedItems = clearUserBuffer(chatId, senderId);
        if (clearedItems.length > 0) {
          log('⏰', `Auto-cleared ${clearedItems.length} item(s) for user ...${shortId} after timeout`);
        }
      }
      chatTimeoutMap.delete(senderId);
    } catch (err) {
      log('❌', `Timeout callback error: ${err.message}`);
    }
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

  // 🔧 LIMIT PREMATURE EXPIRY: Store up to 1000 contexts to safely cover 12h
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
  // 🔧 BUMPED to 2000 so the bot doesn't forget its own messages prematurely
  if (ids.size > 2000) {
    const arr = Array.from(ids);
    arr.slice(0, 1000).forEach(id => ids.delete(id));
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

    const batchSize = 3;
    const inputFps = targetFps * batchSize;

    const videoFilter = `fps=${inputFps},thumbnail=${batchSize}`;

    log('🎬', `Smart Extract: Target ${targetFps}fps (Input ${inputFps}fps, Batch ${batchSize})`);

    ffmpeg(inputPath)
      .outputOptions([
        `-vf ${videoFilter}`,
        '-vsync 0',
        '-q:v 2'
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
            fs.unlinkSync(path);
            return buffer.toString('base64');
          });

          fs.unlinkSync(inputPath);
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

// ======================================================================
// 🔗 MEDIA VIEWER ROUTE
// ======================================================================
app.get('/view/:viewerId', (req, res) => {
  const { viewerId } = req.params;
  const entry = mediaViewerStore.get(viewerId);

  if (!entry) {
    return res.status(404).send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>Link Expired</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #1a1a2e; color: white; }
            .container { text-align: center; padding: 40px; }
            .icon { font-size: 64px; margin-bottom: 20px; }
            h1 { color: #e94560; }
            p { color: #aaa; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">⏰</div>
            <h1>Link Expired or Invalid</h1>
            <p>This media viewer link has expired (12 hour limit) or does not exist.</p>
        </div>
    </body>
    </html>`);
  }

  if (Date.now() >= entry.expiresAt) {
    mediaViewerStore.delete(viewerId);
    return res.status(410).send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>Link Expired</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #1a1a2e; color: white; }
            .container { text-align: center; padding: 40px; }
            .icon { font-size: 64px; margin-bottom: 20px; }
            h1 { color: #e94560; }
            p { color: #aaa; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">⏰</div>
            <h1>Link Expired</h1>
            <p>This media viewer link has expired (12 hour limit).</p>
        </div>
    </body>
    </html>`);
  }

  const media = entry.media;
  const remainingMs = entry.expiresAt - Date.now();
  const remainingHours = Math.floor(remainingMs / 3600000);
  const remainingMins = Math.floor((remainingMs % 3600000) / 60000);

  let mediaHtml = '';
  media.forEach((m, index) => {
    const caption = m.caption ? `<div class="caption">${m.caption.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>` : '';
    const fileName = m.fileName ? `<div class="filename">${m.fileName.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>` : '';

    if (m.type === 'image') {
      mediaHtml += `
        <div class="media-item">
            <div class="media-index">#${index + 1} — Image</div>
            ${caption}${fileName}
            <img src="data:${m.mimeType};base64,${m.data}" alt="Source Image ${index + 1}" loading="lazy" onclick="openFullscreen(this)">
        </div>`;
    } else if (m.type === 'pdf') {
      mediaHtml += `
        <div class="media-item">
            <div class="media-index">#${index + 1} — PDF</div>
            ${caption}${fileName}
            <iframe src="data:application/pdf;base64,${m.data}" class="pdf-frame"></iframe>
            <a href="data:application/pdf;base64,${m.data}" download="${m.fileName || 'document.pdf'}" class="download-btn">⬇️ Download PDF</a>
        </div>`;
    } else if (m.type === 'audio' || m.type === 'voice') {
      mediaHtml += `
        <div class="media-item">
            <div class="media-index">#${index + 1} — ${m.type === 'voice' ? 'Voice Note' : 'Audio'}</div>
            ${caption}${fileName}
            <audio controls src="data:${m.mimeType};base64,${m.data}" style="width:100%;"></audio>
        </div>`;
    } else if (m.type === 'video') {
      mediaHtml += `
        <div class="media-item">
            <div class="media-index">#${index + 1} — Video</div>
            ${caption}${fileName}
            <video controls src="data:${m.mimeType};base64,${m.data}" style="width:100%; max-height:500px;"></video>
        </div>`;
    }
  });

  res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>Source Media Viewer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f1a;
                color: #e0e0e0;
                padding: 10px;
            }
            .header {
                text-align: center;
                padding: 20px 10px;
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 12px;
                margin-bottom: 15px;
                border: 1px solid #333;
            }
            .header h1 { font-size: 20px; color: #25D366; margin-bottom: 8px; }
            .header .meta { font-size: 12px; color: #888; }
            .header .expiry { font-size: 11px; color: #e94560; margin-top: 5px; }
            .media-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 12px;
            }
            .media-item {
                background: #1a1a2e;
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid #2a2a4a;
                transition: transform 0.2s;
            }
            .media-item:hover { transform: scale(1.01); border-color: #25D366; }
            .media-item img {
                width: 100%;
                display: block;
                cursor: pointer;
                transition: opacity 0.2s;
            }
            .media-item img:hover { opacity: 0.9; }
            .media-index {
                padding: 8px 12px;
                font-size: 11px;
                font-weight: 600;
                color: #25D366;
                background: #0f0f1a;
                border-bottom: 1px solid #2a2a4a;
            }
            .caption {
                padding: 6px 12px;
                font-size: 12px;
                color: #ccc;
                background: #16213e;
                font-style: italic;
            }
            .filename {
                padding: 4px 12px;
                font-size: 11px;
                color: #888;
            }
            .pdf-frame {
                width: 100%;
                height: 500px;
                border: none;
            }
            .download-btn {
                display: block;
                text-align: center;
                padding: 10px;
                background: #25D366;
                color: #000;
                text-decoration: none;
                font-weight: 600;
                font-size: 13px;
            }
            .download-btn:hover { background: #1da851; }

            .fullscreen-overlay {
                display: none;
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                background: rgba(0,0,0,0.95);
                z-index: 9999;
                justify-content: center;
                align-items: center;
                cursor: zoom-out;
            }
            .fullscreen-overlay.active { display: flex; }
            .fullscreen-overlay img {
                max-width: 95vw;
                max-height: 95vh;
                object-fit: contain;
            }

            @media (max-width: 600px) {
                .media-grid { grid-template-columns: 1fr; }
                body { padding: 5px; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Source Media Viewer</h1>
            <div class="meta">${media.length} file(s) • Created ${new Date(entry.createdAt).toLocaleString()}</div>
            <div class="expiry">⏰ Expires in ${remainingHours}h ${remainingMins}m</div>
        </div>

        <div class="media-grid">
            ${mediaHtml}
        </div>

        <div class="fullscreen-overlay" id="fsOverlay" onclick="closeFullscreen()">
            <img id="fsImage" src="" alt="Fullscreen">
        </div>

        <script>
            function openFullscreen(img) {
                const overlay = document.getElementById('fsOverlay');
                const fsImg = document.getElementById('fsImage');
                fsImg.src = img.src;
                overlay.classList.add('active');
            }
            function closeFullscreen() {
                document.getElementById('fsOverlay').classList.remove('active');
            }
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') closeFullscreen();
            });
        </script>
    </body>
    </html>`);
});

app.get('/media/:viewerId/:index', (req, res) => {
  const { viewerId, index } = req.params;
  const idx = parseInt(index);
  const entry = mediaViewerStore.get(viewerId);

  if (!entry || Date.now() >= entry.expiresAt) {
    return res.status(404).send('Not found or expired');
  }

  if (isNaN(idx) || idx < 0 || idx >= entry.media.length) {
    return res.status(404).send('Invalid index');
  }

  const m = entry.media[idx];
  const buffer = Buffer.from(m.data, 'base64');
  res.setHeader('Content-Type', m.mimeType);
  res.setHeader('Content-Length', buffer.length);
  res.send(buffer);
});
// ======================================================================

app.get('/', (req, res) => {
  let stats = { users: 0, images: 0, pdfs: 0, audio: 0, video: 0, texts: 0, total: 0 };
  for (const[chatId, _] of chatMediaBuffers) {
    const s = getTotalBufferStats(chatId);
    stats.users += s.users;
    stats.images += s.images;
    stats.pdfs += s.pdfs;
    stats.audio += s.audio;
    stats.video += s.video;
    stats.texts += s.texts;
    stats.total += s.total;
  }

  const isForceReset = process.env.FORCE_RESET === 'true';

  let html = `
    <!DOCTYPE html>
    <html>
    <head>
        <title>WhatsApp Patient Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="${qrCodeDataURL ? 30 : 5}">
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
            .warning { background: #ff4444; color: white; border: 2px solid #ff0000; animation: blink 2s infinite; }
            @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.8; } 100% { opacity: 1; } }
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📱 WhatsApp Patient Bot</h1>
            <p class="subtitle">Medical Clinical Profile Generator</p>
            <div class="db-status ${mongoConnected ? 'db-connected' : 'db-disconnected'}">
                ${mongoConnected ? '🗄 MongoDB Connected' : '⚠️ MongoDB Not Connected'}
            </div>
            <div>ℹ️ API Keys Loaded: ${CONFIG.API_KEYS.length}</div>
    `;

  if (isForceReset) {
    html += `
            <div class="status warning">🚨 FORCE_RESET IS ACTIVE 🚨</div>
            <p><strong>IMPORTANT:</strong> Go to your Render Dashboard and delete the <code>FORCE_RESET</code> variable (or set to false), then restart. Otherwise, the bot will wipe its memory every time it restarts!</p>
    `;
  }

  if (isConnected) {
    html += `
            <div class="status connected">✅ UNIVERSAL MODE ACTIVE</div>
            <p>Bot is active for all incoming messages (Private & Group)</p>
            <div class="stats">
                <div class="stat"><div class="stat-value">${stats.users}</div><div class="stat-label">Active Chats</div></div>
                <div class="stat"><div class="stat-value">${stats.total}</div><div class="stat-label">Buffered</div></div>
                <div class="stat"><div class="stat-value">${processedCount}</div><div class="stat-label">✅ Done</div></div>
                <div class="stat"><div class="stat-value">${mediaViewerStore.size}</div><div class="stat-label">🔗 Viewers</div></div>
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
                    <strong>🔗 Source Viewer:</strong> Each response includes a link to view source media (12h expiry)<br>
                    <strong>🔧 Auto-Heal:</strong> Signal session key auto-repair active<br>
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
    activeViewers: mediaViewerStore.size,
    decryptFailures: decryptFailTimestamps.length,
    healingInProgress: isHealingInProgress,
    pendingRetries: pendingEmptyMessages.size,
    timestamp: new Date().toISOString()
  });
});

app.listen(PORT, () => {
  log('🌐', `Web server running on port ${PORT}`);
});

// ======================================================================
// 🔗 HELPER: Get the base URL for viewer links
// ======================================================================
function getBaseUrl() {
  if (process.env.RENDER_EXTERNAL_URL) {
    return process.env.RENDER_EXTERNAL_URL;
  }
  return `http://localhost:${PORT}`;
}
// ======================================================================

// ======================================================================
// 🔗 HELPER: Parse JSON block from AI response
// ======================================================================
function parseJsonFromResponse(responseText) {
  const jsonMatch = responseText.match(/<<JSON>>(.*?)<<JSON>>/s);
  if (jsonMatch && jsonMatch[1]) {
    try {
      return JSON.parse(jsonMatch[1].trim());
    } catch (e) {
      log('⚠️', `Failed to parse JSON block from response: ${e.message}`);
      return null;
    }
  }
  return null;
}

function stripJsonFromResponse(responseText) {
  return responseText.replace(/\n*<<JSON>>.*?<<JSON>>\n*/s, '').trim();
}

function formatJsonBlock(jsonData) {
  if (!jsonData) return '';
  const age = jsonData.age || 'unknown';
  const sex = jsonData.sex || 'unknown';
  const study = jsonData.study || 'Not mentioned';
  const brief = jsonData.brief || '';
  return `\n\n📋 *Quick Reference:*\n• Age: ${age}\n• Sex: ${sex}\n• Study: ${study}\n• Brief: ${brief}`;
}

// 🔧 FIX: Made all usernames correctly clickable, explicitly maps the JID @string into text
function formatSenderContact(senderId, senderName) {
  if (!senderId) return { text: '', mentionId: null };
  const userString = senderId.split('@')[0];
  
  // Use @userString and include senderName to be readable.
  // Using the strict "@numericalId" text natively signals the WhatsApp client to hook it to the mentionId.
  const namePart = senderName && senderName !== userString ? ` (${senderName})` : '';
  const displayString = `\n\n👤 *Sent by:* @${userString}${namePart}`;
  
  return { text: displayString, mentionId: senderId };
}

// ======================================================================
// 💬 HELPER: Footer text for group chat bot messages
// ======================================================================
const GROUP_REPLY_FOOTER = `\n\n_💬 If you want to ask anything about this patient, select and reply to this message_`;

// ======================================================================

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

    if (!mongoConnected && CONFIG.MONGODB_URI) {
      log('⚠️', 'MongoDB appears disconnected. Attempting to reconnect...');
      await connectMongoDB();
    }

    // 🟢 NEW: FORCE RESET LOGIC
    if (mongoConnected && process.env.FORCE_RESET === 'true') {
      log('🚨', '╔══════════════════════════════════════════╗');
      log('🚨', '║ FORCE_RESET IS TRUE! WIPING DATABASE...  ║');
      log('🚨', '╚══════════════════════════════════════════╝');
      try {
        if (!SessionModel) {
          SessionModel = mongoose.model('Session', sessionSchema);
        }
        await SessionModel.deleteMany({});
        log('✅', 'Database successfully wiped! A new QR code will be generated.');
      } catch (err) {
        log('⚠️', `Wipe error (collection might not exist yet): ${err.message}`);
      }
    } else if (!startupHealDone && mongoConnected) {
      // ONE-TIME STARTUP HEAL — Nuke stale session keys on first boot
      if (!SessionModel) {
        SessionModel = mongoose.model('Session', sessionSchema);
      }
      log('🔧', '╔══════════════════════════════════════════╗');
      log('🔧', '║ STARTUP HEAL: Cleaning session keys...  ║');
      log('🔧', '╚══════════════════════════════════════════╝');
      const deleted = await nukeSessionKeysFromMongo();
      log('🔧', ` Startup heal complete. Removed ${deleted} stale keys.`);
      startupHealDone = true;
    }

    let state, saveCreds, clearAll, clearSessionKeys;

    if (mongoConnected) {
      try {
        const mongoAuth = await useMongoDBAuthState();
        state = mongoAuth.state;
        saveCreds = mongoAuth.saveCreds;
        clearAll = mongoAuth.clearAll;
        clearSessionKeys = mongoAuth.clearSessionKeys;
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
      clearSessionKeys = async () => {
        try {
          if (fs.existsSync(authPath)) {
            const files = fs.readdirSync(authPath);
            let cleared = 0;
            for (const file of files) {
              if (file !== 'creds.json') {
                fs.unlinkSync(join(authPath, file));
                cleared++;
              }
            }
            log('🗑', `Cleared ${cleared} session key files (kept creds.json)`);
          }
        } catch(e) {
          log('❌', `File session key clear error: ${e.message}`);
        }
      };
      log('📁', 'Using file-based auth (session will be lost on restart)');
    }

    authState = { state, saveCreds, clearAll, clearSessionKeys };

    let version;
    try {
      const v = await fetchLatestBaileysVersion();
      version = v.version;
      log('📱', `Using WA version: ${version.join('.')}`);
    } catch (e) {
      version =[2, 3000, 1015901307];
      log('⚠️', 'Using fallback WA version');
    }

    botStatus = 'Connecting...';

    const baileysLogger = pino({ level: 'silent' });

    // Clean up old listeners before creating a new socket reference
    if (sock) {
      try { sock.ev.removeAllListeners(); } catch (e) {}
    }

    sock = makeWASocket({
      version,
      auth: state,
      logger: baileysLogger,
      browser:['WhatsApp-Bot', 'Chrome', '120.0.0'],
      markOnlineOnConnect: false,
      syncFullHistory: false,
      retryRequestDelayMs: 2000,
      getMessage: async (key) => {
        return { conversation: '' };
      }
    });

    sock.ev.on('connection.update', async (update) => {
      try {
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

            try {
              if (authState?.clearAll) {
                await authState.clearAll();
              }
            } catch (err) { log('❌', `Clear all error: ${err.message}`); }

            log('🔄', 'Restarting with fresh session in 5 seconds...');
            setTimeout(startBot, 5000);
          } else {
            try {
              if (statusCode === 428 || statusCode === 408 || statusCode === 515) {
                log('🔧', `Error ${statusCode} — clearing session keys before reconnect...`);
                await nukeSessionKeysFromMongo();
              }
            } catch (err) { log('❌', `Clear keys error: ${err.message}`); }
            
            log('🔄', `Reconnecting in 5 seconds...`);
            setTimeout(startBot, 5000);
          }

        } else if (connection === 'open') {
          isConnected = true;
          qrCodeDataURL = null;
          botStatus = 'Connected';

          // 🔧 Reset decryption failure counter on successful connection
          decryptFailTimestamps =[];

          log('✅', '🎉 CONNECTED TO WHATSAPP!');

          try {
            if (authState?.saveCreds) {
              await authState.saveCreds();
              log('💾', 'Credentials saved');
            }
          } catch (err) { log('❌', `Cred save error: ${err.message}`); }

          log('🌍', 'Universal Mode: Bot is active for ALL chats.');
          if (CONFIG.GROUPS.CT_SOURCE) log('🏥', 'Monitoring CT Source Group');
          if (CONFIG.GROUPS.MRI_SOURCE) log('🏥', 'Monitoring MRI Source Group');
        }
      } catch (err) {
        log('❌', `Connection update handling error: ${err.message}`);
      }
    });

    sock.ev.on('creds.update', async () => {
      try {
        if (authState?.saveCreds) {
          await authState.saveCreds();
        }
      } catch (err) { log('❌', `Creds update error: ${err.message}`); }
    });

    sock.ev.on('messages.upsert', async ({ messages, type }) => {
      try {
        if (type !== 'notify') return;

        for (const msg of messages) {
          if (msg.key.fromMe) continue;

          const msgId = msg.key.id;

          // 🔧 DECRYPTION FAILURE DETECTION & UNIVERSAL RETRY LOGIC
          if (!msg.message || Object.keys(msg.message).length === 0) {
            const chatId = msg.key.remoteJid;
            if (chatId && chatId !== 'status@broadcast') {
              // 🔄 ALL CHATS: Queue for retry to handle multi-device sync delays & transient decryption fails
              if (msgId) {
                processedMessageIds.delete(msgId); // ALLOW IT TO BE PROCESSED ONCE CONTENT ARRIVES
                if (!pendingEmptyMessages.has(msgId)) {
                  log('⏳', `Empty message body from ${chatId} — queuing for retry (msg: ${msgId.substring(0, 8)}...)`);
                  pendingEmptyMessages.set(msgId, {
                    msg: msg,
                    retryCount: 0,
                    chatId: chatId,
                    timestamp: Date.now()
                  });
                  scheduleEmptyMessageRetry(msgId);
                }
              }
              trackDecryptionFailure(); // Trigger auto-heal threshold if failures persist
            }
            continue;
          }

          // 🆔 DEDUPLICATION: Skip if we've already processed this message ID
          if (msgId && isMessageAlreadyProcessed(msgId)) {
            log('🔁', `Skipping duplicate message ${msgId.substring(0, 8)}...`);
            continue;
          }

          // Check if this was previously an empty message that now has body in upsert
          if (msgId && pendingEmptyMessages.has(msgId)) {
            pendingEmptyMessages.delete(msgId);
            log('✅', `Pending empty message ${msgId.substring(0, 8)}... now has content!`);
          }

          try {
            await handleMessage(msg);
          } catch (error) {
            log('❌', `Message handling error: ${error.message}`);
          }
        }
      } catch (err) {
        log('❌', `Messages upsert outer error: ${err.message}`);
      }
    });

    // 🔄 Listen for message updates (content populated after initial delivery)
    sock.ev.on('messages.update', async (updates) => {
      try {
        for (const update of updates) {
          const msgId = update.key?.id;
          if (!msgId) continue;

          // Check if this was a pending empty message that now has content
          if (pendingEmptyMessages.has(msgId) && update.update?.message) {
            const pending = pendingEmptyMessages.get(msgId);
            log('✅', `Message update received for pending empty msg ${msgId.substring(0, 8)}... — now has content!`);

            // Reconstruct the message with the updated content
            const fullMsg = {
              ...pending.msg,
              message: update.update.message
            };

            // Remove from pending
            pendingEmptyMessages.delete(msgId);
            processedMessageIds.delete(msgId); // Ensure deduplicator allows processing

            // Now process it
            if (!fullMsg.key.fromMe && !isMessageAlreadyProcessed(msgId)) {
              try {
                await handleMessage(fullMsg);
              } catch (error) {
                log('❌', `Message handling error (from update): ${error.message}`);
              }
            }
          }
        }
      } catch (err) {
        log('❌', `Messages update outer error: ${err.message}`);
      }
    });

  } catch (error) {
    log('💥', `Bot error: ${error.message}`);
    console.error(error);
    botStatus = 'Error - restarting...';
    setTimeout(startBot, 10000);
  }
}

// ======================================================================
// 🔄 EMPTY MESSAGE RETRY LOGIC (for source group messages)
// ======================================================================
function scheduleEmptyMessageRetry(msgId) {
  const pending = pendingEmptyMessages.get(msgId);
  if (!pending) return;

  const retryDelay = CONFIG.EMPTY_MSG_RETRY_DELAY_MS; // Constant 5s intervals

  setTimeout(async () => {
    try {
      const stillPending = pendingEmptyMessages.get(msgId);
      if (!stillPending) return; // Already resolved via messages.update or upsert

      stillPending.retryCount++;

      if (stillPending.retryCount >= CONFIG.EMPTY_MSG_MAX_RETRIES) {
        log('⚠️', `Empty message ${msgId.substring(0, 8)}... failed after ${CONFIG.EMPTY_MSG_MAX_RETRIES} retries — giving up (chat: ${stillPending.chatId})`);
        pendingEmptyMessages.delete(msgId);
        return;
      }

      log('🔄', `Retry ${stillPending.retryCount}/${CONFIG.EMPTY_MSG_MAX_RETRIES} for empty message ${msgId.substring(0, 8)}...`);

      // Schedule next retry
      scheduleEmptyMessageRetry(msgId);
    } catch (error) {
      log('❌', `Retry error for ${msgId.substring(0, 8)}...: ${error.message}`);
      pendingEmptyMessages.delete(msgId);
    }
  }, retryDelay);
}

// Periodic cleanup for stale pending messages (older than 2 minutes)
setInterval(() => {
  const now = Date.now();
  let cleaned = 0;
  for (const [msgId, pending] of pendingEmptyMessages) {
    if (now - pending.timestamp > 120000) { // 2 minutes
      pendingEmptyMessages.delete(msgId);
      cleaned++;
    }
  }
  if (cleaned > 0) {
    log('🧹', `Cleaned ${cleaned} stale pending empty message(s)`);
  }
}, 60000);
// ======================================================================

// 🟢 FIX: Helper to unwrap nested messages robustly
const unwrapMessage = (m) => {
  if (!m) return null;
  if (m.message) return unwrapMessage(m.message); // Some messages are nested in .message
  if (m.viewOnceMessage?.message) return unwrapMessage(m.viewOnceMessage.message);
  if (m.viewOnceMessageV2?.message) return unwrapMessage(m.viewOnceMessageV2.message);
  if (m.viewOnceMessageV2Extension?.message) return unwrapMessage(m.viewOnceMessageV2Extension.message);
  if (m.ephemeralMessage?.message) return unwrapMessage(m.ephemeralMessage.message);
  if (m.documentWithCaptionMessage?.message) return unwrapMessage(m.documentWithCaptionMessage.message);
  return m;
};

// 🟢 FIX: Helper to accurately determine the message type, ignoring metadata keys
function getMessageType(content) {
  if (!content) return null;
  const keys = Object.keys(content);
  const validTypes =[
    'imageMessage',
    'videoMessage',
    'audioMessage',
    'documentMessage',
    'extendedTextMessage',
    'conversation'
  ];
  for (const type of validTypes) {
    if (keys.includes(type)) return type;
  }
  return keys[0];
}

async function handleMessage(msg) {
  const chatId = msg.key.remoteJid;

  if (chatId === 'status@broadcast') return;

  const senderId = getSenderId(msg);
  const senderName = getSenderName(msg);
  const shortId = getShortSenderId(senderId);

  const isGroup = chatId.endsWith('@g.us');
  if (isGroup) {
    log('📋', `Message from group: ${chatId} (Allowed: ALL)`);
  }

  const content = unwrapMessage(msg.message);

  if (!content || Object.keys(content).length === 0) {
    return;
  }

  const messageType = getMessageType(content);

  let quotedMessageId = null;
  // 🟢 FIX: Safe universal contextInfo extraction (fixes missing reply detection)
  let contextInfo = content.extendedTextMessage?.contextInfo ||
                    content.imageMessage?.contextInfo ||
                    content.videoMessage?.contextInfo ||
                    content.audioMessage?.contextInfo ||
                    content.documentMessage?.contextInfo ||
                    content[messageType]?.contextInfo;

  if (contextInfo?.stanzaId) {
    quotedMessageId = contextInfo.stanzaId;

    if (isBotMessage(chatId, quotedMessageId)) {
      log('↩️', `Reply to bot from ${senderName} (...${shortId})`);
      await handleReplyToBot(msg, chatId, quotedMessageId, senderId, senderName, messageType, content);
      return;
    }
  }

  if (messageType === 'imageMessage') {
    log('📷', `Image from ${senderName} (...${shortId})`);

    try {
      const buffer = await downloadMediaMessage(msg, 'buffer', {});
      const caption = content.imageMessage.caption || '';

      const count = addToUserBuffer(chatId, senderId, {
        type: 'image',
        data: buffer.toString('base64'),
        mimeType: content.imageMessage.mimetype || 'image/jpeg',
        caption: caption,
        timestamp: Date.now()
      });

      if (caption) {
        log('💬', ` └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
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
      const caption = content.videoMessage.caption || '';
      const mimeType = content.videoMessage.mimetype || 'video/mp4';

      const count = addToUserBuffer(chatId, senderId, {
        type: 'video',
        data: buffer.toString('base64'),
        mimeType: mimeType,
        caption: caption,
        duration: content.videoMessage.seconds || 0,
        timestamp: Date.now()
      });

      if (caption) {
        log('💬', ` └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
      }

      log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
      resetUserTimeout(chatId, senderId, senderName);

    } catch (error) {
      log('❌', `Video error: ${error.message}`);
    }
  }
  else if (messageType === 'audioMessage') {
    const isVoice = content.audioMessage.ptt === true;
    const emoji = isVoice ? '🎤' : '🎵';

    log(emoji, `${isVoice ? 'Voice' : 'Audio'} from ${senderName} (...${shortId})`);

    try {
      const buffer = await downloadMediaMessage(msg, 'buffer', {});

      const count = addToUserBuffer(chatId, senderId, {
        type: isVoice ? 'voice' : 'audio',
        data: buffer.toString('base64'),
        mimeType: content.audioMessage.mimetype || 'audio/ogg',
        duration: content.audioMessage.seconds || 0,
        timestamp: Date.now()
      });

      log('📦', `Buffer for ...${shortId}: ${count} item(s)`);
      resetUserTimeout(chatId, senderId, senderName);

    } catch (error) {
      log('❌', `Audio error: ${error.message}`);
    }
  }
  else if (messageType === 'documentMessage') {
    const docMime = content.documentMessage.mimetype || '';
    const fileName = content.documentMessage.fileName || 'document';
    const caption = content.documentMessage.caption || '';

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
          log('💬', ` └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
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
          log('💬', ` └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
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
          log('💬', ` └─ Caption: "${caption.substring(0, 50)}${caption.length > 50 ? '...' : ''}"`);
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
    const text = (content.conversation || content.extendedTextMessage?.text || '').trim();

    if (!text) return;

    // CHECK FOR TRIGGERS
    const isPrimaryTrigger = /^(\.|(\.[1-3]))$/.test(text);
    const isSecondaryTrigger = /^(\.\.|(\.\.[1-3]))$/.test(text);

    if (isPrimaryTrigger || isSecondaryTrigger) {
      log('🔔', `Trigger command "${text}" from ${senderName} (...${shortId})`);

      await new Promise(r => setTimeout(r, 1000));

      const userBufferCount = getUserBufferCount(chatId, senderId);

      if (userBufferCount > 0) {
        clearUserTimeout(chatId, senderId);
        const mediaFiles = clearUserBuffer(chatId, senderId);

        const lastChar = text.slice(-1);
        let targetFps = 3;
        if (!isNaN(parseInt(lastChar))) {
          targetFps = parseInt(lastChar);
        }

        const batches = groupMediaSmartly(mediaFiles);

        if (batches.length > 1) {
          log('🔀', `Manual Trigger: Detected ${batches.length} distinct patient contexts.`);
        }

        for (let i = 0; i < batches.length; i++) {
          const batch = batches[i];
          if (batch.length === 0) continue;

          const modeLabel = isSecondaryTrigger ? 'SECONDARY/CHAINED' : 'PRIMARY';
          log('🤖', `Processing Batch ${i+1}/${batches.length} (${batch.length} items) with FPS=${targetFps}. Mode: ${modeLabel}`);

          await processMedia(chatId, batch, false, null, senderId, senderName, null, targetFps, isSecondaryTrigger);

          if (i < batches.length - 1) {
            await new Promise(r => setTimeout(r, 2000));
          }
        }

      } else {
        if (sock) {
          await sock.sendMessage(chatId, {
            text: `ℹ️ @${senderId.split('@')[0]}, you have no files buffered.\n\nSend files first, then send *.* (Standard) or *..* (Secondary Analysis).\nAdd numbers for video speed (e.g. .2 or ..2)\n\n💡 _Or reply to my previous response to ask questions!_`,
            mentions: [senderId]
          }).catch(() => {});
        }
      }
    }
    else if (text.toLowerCase() === 'help' || text === '?') {
      if (sock) {
        await sock.sendMessage(chatId, {
          text: `🏥 *Clinical Profile Bot*\n\n*Universal Mode Active*\nI work in this chat and any group I'm added to!\n\n*Supported Files:*\n📷 Images, 📄 PDFs, 🎤 Voice, 🎵 Audio, 🎬 Video\n\n*Commands:*\n• *.* - Standard Clinical Profile (Smart 3 FPS)\n• *..* - Secondary Chained Analysis (Profile + Advice)\n• *.1 / ..1* - Process with Smart 1 FPS\n• *.2 / ..2* - Process with Smart 2 FPS\n• *clear* - Clear buffer\n• *status* - Check status\n\n*Reply Feature:*\nReply to my messages to ask questions or provide corrections!\n\n*🔗 Source Viewer:*\nEach response includes a link to view source media (valid 12h)`
        }).catch(() => {});
      }
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

        if (sock) {
          await sock.sendMessage(chatId, {
            text: `🗑 @${senderId.split('@')[0]}, cleared your buffer:\n📷 ${counts.images} image(s)\n📄 ${counts.pdfs} PDF(s)\n🎵 ${counts.audio} audio\n🎬 ${counts.video} video(s)\n💬 ${counts.texts} text(s)`,
            mentions: [senderId]
          }).catch(() => {});
        }
      } else {
        if (sock) {
          await sock.sendMessage(chatId, {
            text: `ℹ️ @${senderId.split('@')[0]}, your buffer is empty.`,
            mentions:[senderId]
          }).catch(() => {});
        }
      }
    }
    else if (text.toLowerCase() === 'status') {
      const stats = getTotalBufferStats(chatId);
      const userCount = getUserBufferCount(chatId, senderId);
      const storedContexts = chatContexts.has(chatId) ? chatContexts.get(chatId).size : 0;

      if (sock) {
        await sock.sendMessage(chatId, {
          text: `📊 *Status*\n\n*Your Buffer:* ${userCount} item(s)\n\n*Chat Total:*\n👥 Active users: ${stats.users}\n📷 Images: ${stats.images}\n📄 PDFs: ${stats.pdfs}\n🎵 Audio: ${stats.audio}\n🎬 Video: ${stats.video}\n💬 Texts: ${stats.texts}\n━━━━━━━━━━\n📦 Total buffered: ${stats.total}\n🧠 Stored contexts: ${storedContexts}\n✅ Processed: ${processedCount}\n🗄 MongoDB: ${mongoConnected ? 'Connected' : 'Not connected'}\n🔑 API Keys: ${CONFIG.API_KEYS.length} available\n🔗 Active Viewers: ${mediaViewerStore.size}\n🔧 Decrypt Fails (1min): ${decryptFailTimestamps.length}/${CONFIG.DECRYPT_FAIL_THRESHOLD}\n⏳ Pending Retries: ${pendingEmptyMessages.size}`
        }).catch(() => {});
      }
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
  else {
    log('📎', `Skipping unhandled/system message type: ${messageType}`);
  }
}

async function handleReplyToBot(msg, chatId, quotedMessageId, senderId, senderName, messageType, content) {
  const storedContext = getStoredContext(chatId, quotedMessageId);
  const shortId = getShortSenderId(senderId);
  const isGroup = chatId.endsWith('@g.us');

  if (!storedContext) {
    // 🔧 FIX #1: Updated expiry message from "30 min limit" to "12 hour limit"
    log('⚠️', `Context expired for ...${shortId}`);
    if (sock) {
      await sock.sendMessage(chatId, {
        text: `⏰ @${senderId.split('@')[0]}, that context has expired (12 hour limit).\n\nPlease send new files and use "." to process.`,
        mentions: [senderId]
      }).catch(() => {});
    }
    return;
  }

  // ======================================================================
  // 🆕 GROUP CHAT REPLY: Exclusively use source media + user text, NO system instruction
  // ======================================================================
  if (isGroup) {
    // 🔧 FIX: Correctly parse question/text from media captions during group replies
    let userQuestion = '';

    if (messageType === 'conversation') {
      userQuestion = (content.conversation || '').trim();
    } else if (messageType === 'extendedTextMessage') {
      userQuestion = (content.extendedTextMessage?.text || '').trim();
    } else if (messageType === 'imageMessage') {
      userQuestion = (content.imageMessage?.caption || '').trim();
    } else if (messageType === 'videoMessage') {
      userQuestion = (content.videoMessage?.caption || '').trim();
    } else if (messageType === 'documentMessage') {
      userQuestion = (content.documentMessage?.caption || '').trim();
    }

    if (!userQuestion) {
      if (sock) {
        await sock.sendMessage(chatId, {
          text: `ℹ️ @${senderId.split('@')[0]}, please type your question as text when replying to the message.`,
          mentions: [senderId]
        }).catch(() => {});
      }
      return;
    }

    log('💬', `Group reply-question from ...${shortId}: "${userQuestion.substring(0, 80)}..."`);

    // Build content parts from the ORIGINAL source documents
    const sourceMedia = storedContext.mediaFiles;
    const contentParts =[];

    for (const media of sourceMedia) {
      if (media.data && media.mimeType && (media.type === 'image' || media.type === 'pdf' || media.type === 'audio' || media.type === 'voice' || media.type === 'video')) {
        contentParts.push({
          inlineData: {
            data: media.data,
            mimeType: media.mimeType
          }
        });
      }
    }

    // EXCLUSIVELY: source documents as inline data + user's question as text — NO system instruction
    const requestContent = contentParts.length > 0
      ? [userQuestion, ...contentParts]
      : [userQuestion];

    try {
      if (sock) await sock.sendPresenceUpdate('composing', chatId).catch(() => {});

      log('🔄', `Group reply (NO system instruction): Sending ${contentParts.length} source doc(s) + question to model for ...${shortId}`);

      // Call Gemini with NO system instruction (null)
      const responseText = await generateGeminiContent(requestContent, null);

      if (sock) await sock.sendPresenceUpdate('paused', chatId).catch(() => {});

      let finalText = responseText.length <= 60000
        ? responseText
        : responseText.substring(0, 60000) + '\n\n_(truncated)_';

      // Add the group reply footer
      finalText += GROUP_REPLY_FOOTER;

      if (sock) {
        const sentMessage = await sock.sendMessage(chatId, {
          text: finalText,
          mentions: [senderId]
        });

        if (sentMessage?.key?.id) {
          const messageId = sentMessage.key.id;
          trackBotMessage(chatId, messageId);
          // Store context with the SAME source media files so further replies also work
          storeContext(chatId, messageId, sourceMedia, responseText, senderId);
          log('💾', `Group reply context stored for ...${shortId}`);
        }
      }

      processedCount++;
      log('📤', `Group reply sent for ...${shortId}`);

    } catch (error) {
      log('❌', `Group reply error for ...${shortId}: ${error.message}`);
      if (sock) {
        await sock.sendMessage(chatId, {
          text: `❌ @${senderId.split('@')[0]}, error processing your question:\n_${error.message}_\n\nPlease try again later.`,
          mentions: [senderId]
        }).catch(() => {});
      }
    }

    return;
  }
  // ======================================================================
  // END GROUP CHAT REPLY — below is the original private chat reply logic
  // ======================================================================

  const newContent =[];
  let userTextInput = '';

  if (messageType === 'conversation') {
    const text = content.conversation || '';
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
    const text = content.extendedTextMessage?.text || '';
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
      const caption = content.imageMessage.caption || '';

      newContent.push({
        type: 'image',
        data: buffer.toString('base64'),
        mimeType: content.imageMessage.mimetype || 'image/jpeg',
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
      const caption = content.videoMessage.caption || '';

      newContent.push({
        type: 'video',
        data: buffer.toString('base64'),
        mimeType: content.videoMessage.mimetype || 'video/mp4',
        caption: caption,
        duration: content.videoMessage.seconds || 0,
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
      const isVoice = content.audioMessage.ptt === true;

      newContent.push({
        type: isVoice ? 'voice' : 'audio',
        data: buffer.toString('base64'),
        mimeType: content.audioMessage.mimetype || 'audio/ogg',
        duration: content.audioMessage.seconds || 0,
        timestamp: Date.now(),
        isFollowUp: true
      });
      log(isVoice ? '🎤' : '🎵', `Follow-up audio from ...${shortId}`);
    } catch (error) {
      log('❌', `Audio error: ${error.message}`);
    }
  }
  else if (messageType === 'documentMessage') {
    const docMime = content.documentMessage.mimetype || '';
    const fileName = content.documentMessage.fileName || 'document';
    const caption = content.documentMessage.caption || '';

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
    if (sock) {
      await sock.sendMessage(chatId, {
        text: `ℹ️ @${senderId.split('@')[0]}, please include text, image, PDF, audio, or video in your reply.`,
        mentions: [senderId]
      }).catch(() => {});
    }
    return;
  }

  const isUserQuestion = isQuestion(userTextInput);

  const combinedMedia =[...storedContext.mediaFiles, ...newContent];

  log('🔄', `Regenerating for ...${shortId}: ${storedContext.mediaFiles.length} original + ${newContent.length} new (isQuestion: ${isUserQuestion})`);

  await processMedia(chatId, combinedMedia, true, storedContext.response, senderId, senderName, userTextInput);
}

// 🟢 NEW: Unified Helper Function for Gemini API Calls with Fallback logic
async function generateGeminiContent(requestContent, systemInstruction) {
  const keys = CONFIG.API_KEYS;
  if (keys.length === 0) {
    throw new Error('No API keys configured!');
  }

  let lastErrorMsg = '';

  const safetySettings =[
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  ];

  // Helper function to try sending a request across all keys with a specified model
  const tryModel = async (modelName, useThinking) => {
    for (let i = 0; i < keys.length; i++) {
      try {
        if (i > 0) {
          log('⚠️', `Waiting 2s before retrying with Backup Key #${i + 1} (${modelName})...`);
          await new Promise(resolve => setTimeout(resolve, 2000));
        }

        const genAI = new GoogleGenerativeAI(keys[i]);
        const modelConfig = {
          model: modelName,
          safetySettings: safetySettings
        };
        if (systemInstruction) {
          modelConfig.systemInstruction = systemInstruction;
        }
        if (useThinking) {
          modelConfig.generationConfig = {
            thinkingConfig: { thinkingLevel: 'HIGH' }
          };
        }

        const model = genAI.getGenerativeModel(modelConfig);
        const result = await model.generateContent(requestContent);
        const responseText = result.response.text();

        if (!responseText) {
          const feedback = JSON.stringify(result.response.promptFeedback || {});
          throw new Error(`Empty response from API (Safety/Filter/Glitch). Feedback: ${feedback}`);
        }

        return responseText;

      } catch (error) {
        lastErrorMsg = error.message;
        log('❌', `Key #${i + 1} (${modelName}) failed: ${error.message}`);
      }
    }
    return null; // All keys failed for this model
  };

  // --- 1. Loop through keys using Primary Model ---
  let responseText = await tryModel(CONFIG.GEMINI_MODEL, false);
  if (responseText) {
    return responseText;
  }

  log('⚠️', `All keys failed for primary model. Falling back to gemini-3.1-flash-lite-preview with HIGH thinking...`);

  // --- 2. Loop through keys using Fallback Lite Model (High Thinking) ---
  responseText = await tryModel('gemini-3.1-flash-lite-preview', true);
  if (responseText) {
    // Append fallback note string natively so it's always included automatically
    return responseText + '\n\n_{fallback model gemini-3.1-flash-lite used}_';
  }

  // --- 3. Both models failed across all keys ---
  throw new Error(`All ${keys.length} API keys failed for both primary and fallback models. Last error: ${lastErrorMsg}`);
}

async function processMedia(chatId, mediaFiles, isFollowUp = false, previousResponse = null, senderId, senderName, userTextInput = null, targetFps = 3, isSecondaryMode = false, targetChatId = null, retryAttempt = 0) {
  const shortId = getShortSenderId(senderId);
  const destinationChatId = targetChatId || chatId;
  const isDestinationGroup = destinationChatId.endsWith('@g.us');

  try {
    const counts = { images: 0, pdfs: 0, audio: 0, video: 0, texts: 0, followUps: 0 };
    const textContents =[];
    const captions = [];
    const binaryMedia = [];
    const followUpTexts =[];

    // === VIDEO PRE-PROCESSING START ===
    const processedMedia =[];

    for (const m of mediaFiles) {
      if (m.type === 'video') {
        log('🎬', `Processing video for ...${shortId} (Target FPS: ${targetFps})`);
        try {
          const videoBuffer = Buffer.from(m.data, 'base64');
          const frames = await extractFramesFromVideo(videoBuffer, targetFps);
          log('📸', `Extracted ${frames.length} smart frames from video`);

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

    const contentParts =[];
    binaryMedia.forEach(media => {
      contentParts.push({
        inlineData: {
          data: media.data,
          mimeType: media.mimeType
        }
      });
    });

    let promptParts =[];
    if (counts.images > 0) promptParts.push(`${counts.images} image(s)`);
    if (counts.pdfs > 0) promptParts.push(`${counts.pdfs} PDF document(s)`);
    if (counts.audio > 0) promptParts.push(`${counts.audio} audio/voice recording(s)`);
    if (counts.video > 0) promptParts.push(`${counts.video} video file(s)`);

    const allOriginalText = [...captions, ...textContents];
    let promptText = '';

    if (isFollowUp && previousResponse) {
      const isUserQuestion = userTextInput ? isQuestion(userTextInput) : false;

      if (isUserQuestion) {
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
- Exclude patient name, age, gender
- Include the <<JSON>> block after the profile`;
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
      requestContent =[promptText, ...contentParts];
    } else {
      requestContent = [promptText];
    }

    // 🔗 STORE SOURCE MEDIA FOR VIEWER
    const viewerId = storeMediaForViewer(mediaFiles);
    const viewerUrl = viewerId ? `${getBaseUrl()}/view/${viewerId}` : null;

    // --- STEP 1: Generate Primary Clinical Profile ---
    log('🔄', `Generating Primary Response (Secondary Mode: ${isSecondaryMode})...`);
    const rawPrimaryResponse = await generateGeminiContent(requestContent, CONFIG.SYSTEM_INSTRUCTION);

    // Parse JSON from response
    const jsonData = parseJsonFromResponse(rawPrimaryResponse);
    const primaryResponseText = stripJsonFromResponse(rawPrimaryResponse);

    if (isSecondaryMode && !isFollowUp) {
      // Build mentions array for this message
      const step1Mentions = [senderId];
      let step1Text = `📝 *Clinical Profile (Step 1):*\n\n${primaryResponseText}`;
      if (jsonData) {
        step1Text += formatJsonBlock(jsonData);
      }
      if (viewerUrl) {
        step1Text += `\n\n🔗 *Source Media:* ${viewerUrl}`;
      }
      // Add sender contact for auto-group routing using @mention
      if (targetChatId) {
        const senderContact = formatSenderContact(senderId, senderName);
        step1Text += senderContact.text;
        if (senderContact.mentionId && !step1Mentions.includes(senderContact.mentionId)) {
          step1Mentions.push(senderContact.mentionId);
        }
      }
      // Add group reply footer if destination is a group
      if (isDestinationGroup) {
        step1Text += GROUP_REPLY_FOOTER;
      }

      if (sock) {
        await sock.sendMessage(destinationChatId, {
          text: step1Text,
          mentions: step1Mentions
        });
      }
      log('📤', `Sent Primary (Step 1) to ...${shortId}`);

      // --- STEP 2: Generate Secondary Analysis ---
      log('🔄', `Generating Secondary Analysis...`);

      const secondaryPrompt = `${SECONDARY_TRIGGER_PROMPT}

=== CLINICAL PROFILE ===
${primaryResponseText}
=== END PROFILE ===`;

      const secondaryRequestContent = [secondaryPrompt];
      const secondaryResponseText = await generateGeminiContent(secondaryRequestContent, SECONDARY_SYSTEM_INSTRUCTION);

      // Build mentions array for secondary message
      const step2Mentions = [senderId];
      let finalSecondaryText = `🧠 *Secondary Analysis (Step 2):*\n\n${secondaryResponseText}`;
      if (viewerUrl) {
        finalSecondaryText += `\n\n🔗 *Source Media:* ${viewerUrl}`;
      }
      if (targetChatId) {
        const senderContact = formatSenderContact(senderId, senderName);
        finalSecondaryText += senderContact.text;
        if (senderContact.mentionId && !step2Mentions.includes(senderContact.mentionId)) {
          step2Mentions.push(senderContact.mentionId);
        }
      }
      // Add group reply footer if destination is a group
      if (isDestinationGroup) {
        finalSecondaryText += GROUP_REPLY_FOOTER;
      }

      processedCount++;

      console.log('\n' + '═'.repeat(60));
      console.log(`👤 User: ${senderName} (...${shortId})`);
      console.log(`📊 CHAINED ANALYSIS COMPLETE`);
      console.log(`⏰ ${new Date().toLocaleString()}`);
      console.log('═'.repeat(60));
      console.log(finalSecondaryText);
      console.log('═'.repeat(60) + '\n');

      if (sock) {
        await sock.sendPresenceUpdate('composing', destinationChatId).catch(() => {});
        await new Promise(resolve => setTimeout(resolve, 2000));
        await sock.sendPresenceUpdate('paused', destinationChatId).catch(() => {});

        const sentMessage = await sock.sendMessage(destinationChatId, {
          text: finalSecondaryText,
          mentions: step2Mentions
        });

        if (sentMessage?.key?.id) {
          const messageId = sentMessage.key.id;
          trackBotMessage(destinationChatId, messageId);
          storeContext(destinationChatId, messageId, mediaFiles, secondaryResponseText, senderId);
          log('💾', `Secondary Context stored for ...${shortId}`);
        }
      }
      log('📤', `Sent Secondary (Step 2) to target!`);
      return;
    }

    // --- NORMAL PRIMARY MODE or FOLLOW-UP HANDLING ---

    if (!primaryResponseText || primaryResponseText.trim() === '') {
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
    if (jsonData) console.log(`JSON: ${JSON.stringify(jsonData)}`);
    console.log('═'.repeat(60) + '\n');

    if (sock) {
      await sock.sendPresenceUpdate('composing', destinationChatId).catch(() => {});
      const delay = Math.floor(Math.random() * (CONFIG.TYPING_DELAY_MAX - CONFIG.TYPING_DELAY_MIN)) + CONFIG.TYPING_DELAY_MIN;
      await new Promise(resolve => setTimeout(resolve, delay));
      await sock.sendPresenceUpdate('paused', destinationChatId).catch(() => {});
    }

    let finalResponseText = primaryResponseText.length <= 60000
      ? primaryResponseText
      : primaryResponseText.substring(0, 60000) + '\n\n_(truncated)_';

    if (targetChatId && allOriginalText.length > 0) {
      const captionHeader = allOriginalText.map(t => t.replace(/^\[.*?\]:\s*/, '')).join('\n');
      if (captionHeader.trim().length > 0) {
        finalResponseText = `${captionHeader}\n\n${finalResponseText}`;
      }
    }

    // Append JSON block
    if (jsonData) {
      finalResponseText += formatJsonBlock(jsonData);
    }

    // 🔗 Append viewer URL to response
    if (viewerUrl) {
      finalResponseText += `\n\n🔗 *Source Media:* ${viewerUrl}`;
    }

    // Build mentions array
    const finalMentions = [senderId];

    // 👤 Append sender contact for auto-group routing using @mention
    if (targetChatId) {
      const senderContact = formatSenderContact(senderId, senderName);
      finalResponseText += senderContact.text;
      if (senderContact.mentionId && !finalMentions.includes(senderContact.mentionId)) {
        finalMentions.push(senderContact.mentionId);
      }
    }

    // 💬 Append group reply footer if destination is a group
    if (isDestinationGroup) {
      finalResponseText += GROUP_REPLY_FOOTER;
    }

    if (sock) {
      const sentMessage = await sock.sendMessage(destinationChatId, {
        text: finalResponseText,
        mentions: finalMentions
      });

      if (sentMessage?.key?.id) {
        const messageId = sentMessage.key.id;
        trackBotMessage(destinationChatId, messageId);
        storeContext(destinationChatId, messageId, mediaFiles, primaryResponseText, senderId);
        log('💾', `Context stored for ...${shortId}`);
      }
    }

    log('📤', `Sent to target/chat!`);

  } catch (error) {
    log('❌', `Error for ...${shortId}: ${error.message}`);
    console.error(error);

    try {
      if (retryAttempt === 0) {
        log('⏳', `Generation failed. Scheduling retry in 5 mins for ...${shortId}`);

        if (sock) {
          await sock.sendPresenceUpdate('composing', destinationChatId).catch(() => {});
          await new Promise(r => setTimeout(r, 1000));

          await sock.sendMessage(destinationChatId, {
            text: `⚠️ *High Traffic / Network Alert*\n\nThe AI model is currently overloaded/unstable. I have queued your request and will *automatically retry in 5 minutes*.\n\nPlease do not resend the files.`,
            mentions: [senderId]
          }).catch(e => log('❌', `Retry message failed: ${e.message}`));
        }

        setTimeout(async () => {
          log('🔄', `Executing 5-minute retry for ...${shortId}`);
          try {
            await processMedia(chatId, mediaFiles, isFollowUp, previousResponse, senderId, senderName, userTextInput, targetFps, isSecondaryMode, targetChatId, 1);
          } catch (retryErr) {
            log('❌', `Retry execution failed: ${retryErr.message}`);
          }
        }, 300000);

        return;
      }

      if (sock) {
        await sock.sendPresenceUpdate('composing', destinationChatId).catch(() => {});
        await new Promise(r => setTimeout(r, 1500));

        await sock.sendMessage(destinationChatId, {
          text: `❌ @${senderId.split('@')[0]}, error processing your request:\n_${error.message}_\n\nPlease try again later.`,
          mentions:[senderId]
        }).catch(e => log('❌', `Error message failed: ${e.message}`));
      }
    } catch (fallbackError) {
      log('❌', `Fallback block error: ${fallbackError.message}`);
    }
  }
}

console.log('\n╔══════════════════════════════════════════════════════════╗');
console.log('║         WhatsApp Clinical Profile Bot v3.6              ║');
console.log('║                                                        ║');
console.log('║  📷 Images 📄 PDFs 🎤 Voice 🎵 Audio 🎬 Video 💬 Text ║');
console.log('║                                                        ║');
console.log('║  🌍 UNIVERSAL MODE: Works in any chat (Group or Private)║');
console.log('║  🔄 AUTO-GROUPS: Monitors Source -> Sends to Target (60s)║');
console.log('║  🔀 SMART BATCHING: Splits distinct patients automatically║');
console.log('║  🎥 SMART VIDEO: Oversamples & Picks Sharpest Frames   ║');
console.log('║      Use: . (3fps), .2 (2fps), .1 (1fps)               ║');
console.log('║  🧠 SECONDARY ANALYSIS: Use .. (double dot) for Chain  ║');
console.log('║  🔗 SOURCE VIEWER: Each response has a 12h media link  ║');
console.log('║  🔧 AUTO-HEAL: Signal session key auto-repair + 428 fix║');
console.log('║  🆔 DEDUPLICATION: Prevents duplicate message processing║');
console.log('║  📋 JSON SUMMARY: Age/Sex/Study/Brief in every response║');
console.log('║  👤 SENDER CONTACT: @mention tag for source sender     ║');
console.log('║  🔄 EMPTY MSG RETRY: Auto-retry empty messages globally║');
console.log('║  ⚡ FALLBACK MODEL: Lite Model Auto-failover w/ HIGH T ║');
console.log('║                                                        ║');
console.log('║  ✨ Per-User Buffers - Each user processed separately  ║');
console.log('║  ↩️ Reply to ask questions OR add context               ║');
console.log('║  🗄 MongoDB Persistent Sessions                        ║');
console.log('║  🔑 Multi-Key Rotation (2hrs) + Failover Active        ║');
console.log('╚══════════════════════════════════════════════════════════╝\n');

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
