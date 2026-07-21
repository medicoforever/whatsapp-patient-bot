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
// 🟢 CONFIGURATION AREA
// ======================================================================

const SECONDARY_SYSTEM_INSTRUCTION = `You are an expert radiologist. When you receive a context, it is mostly about a patient and sometimes they might have been advised with any imaging modality. You analyse that info and then advise regarding that as an expert radiologist what to be seen in that specific imaging modality for that specific patient including various hypothetical imaging findings from common to less common for that patient condition in that specific imaging modality. suppose of you cant indentify thr specific imaging modality in thr given context, you yourself choose the appropriate imaging modality based on the specific conditions context`;

const SECONDARY_TRIGGER_PROMPT = `Here is the Clinical Profile generated from the patient's reports. Please analyze this profile according to your system instructions and provide the final output.`;

// ======================================================================

const CONFIG = {
  API_KEYS: getApiKeys(),
  GEMINI_MODEL: 'gemini-3.6-flash',
  GEMINI_MODEL_FALLBACK_1: 'gemini-3.5-flash',
  GEMINI_MODEL_FALLBACK_2: 'gemini-3-flash-preview',
  GEMINI_MODEL_FALLBACK_3: 'gemini-3.5-flash-lite',
  MONGODB_URI: process.env.MONGODB_URI,

  GROUPS: {
    CT_SOURCE: process.env.GROUP_CT_SOURCE,
    CT_TARGET: process.env.GROUP_CT_TARGET,
    MRI_SOURCE: process.env.GROUP_MRI_SOURCE,
    MRI_TARGET: process.env.GROUP_MRI_TARGET
  },

  MEDIA_TIMEOUT_MS: 300000, // 5 minutes (Standard users)
  AUTO_PROCESS_DELAY_MS: 60000, // 60 seconds (Auto-groups)

  CONTEXT_RETENTION_MS: 12 * 60 * 60 * 1000, // 12 hours
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
  MEDIA_VIEWER_EXPIRY_MS: 12 * 60 * 60 * 1000,
  DECRYPT_FAIL_THRESHOLD: 8,
  DECRYPT_FAIL_WINDOW_MS: 60000,
  EMPTY_MSG_RETRY_DELAY_MS: 3000,
  EMPTY_MSG_MAX_RETRIES: 3,
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
<<JSON>>{"mrn":"<Registration Number/MRN or Not mentioned>","age":"<age or unknown>","sex":"<M/F/unknown>","study":"<imaging study indicated or Not mentioned>","brief":"<very concise reason for scan using abbreviations like H/o, C/o, K/c/o, etc., mentioning duration of symptoms>"}<<JSON>>

Rules for the JSON line:
- mrn: Extract the patient's Medical Record Number (MRN), Registration Number, ID, UID, or IP/OP number from the content. If not found, use "Not mentioned".
- age: Extract patient age from the content. If not found, use "unknown".
- sex: Extract patient sex/gender from the content. Use "M" for male, "F" for female. If not found, use "unknown".
- study: The imaging study that is currently indicated/requested (e.g., "CT Thorax", "MRI Brain", "USG Abdomen"). If not obvious from the content, use "Not mentioned".
- brief: A very short clinical summary using medical abbreviations. Example: "H/o fever and cough for 4 days, SOB for 2 days, K/c/o ILD, Now scan done to r/o infective exacerbation" or "C/o Giddiness for 15 days, slurred speech for 5 days, Right upper limb weakness for 2 days, K/c/o HTN/DM, Now scan done to r/o cerebellar infarct"


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
// 🗄️ PERSISTENT MEDIA BUFFER SCHEMA (MongoDB + RAM Backup)
// ======================================================================
const pendingMediaSchema = new mongoose.Schema({
  chatId: { type: String, required: true, index: true },
  senderId: { type: String, required: true, index: true },
  senderName: String,
  messageId: { type: String, required: true, unique: true },
  type: String,
  data: String, // Base64 content
  mimeType: String,
  caption: String,
  fileName: String,
  processed: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now }
}, { collection: 'whatsapp_pending_media' });

pendingMediaSchema.index({ createdAt: 1 }, { expireAfterSeconds: 86400 * 7 }); // 7 days auto-cleanup

let PendingMediaModel;

function getPendingMediaModel() {
  if (!PendingMediaModel && mongoose.connection.readyState === 1) {
    PendingMediaModel = mongoose.model('PendingMedia', pendingMediaSchema);
  }
  return PendingMediaModel;
}

// Fallback in-memory map if MongoDB is temporarily unavailable
const ramMediaBuffers = new Map();

// ======================================================================
// 🔗 MEDIA VIEWER STORE (In-Memory with 12hr Expiry)
// ======================================================================
const mediaViewerStore = new Map();

function storeMediaForViewer(mediaFiles) {
  const viewerId = crypto.randomBytes(16).toString('hex');
  const expiresAt = Date.now() + CONFIG.MEDIA_VIEWER_EXPIRY_MS;

  const viewableMedia = [];
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
  for (const [id, entry] of mediaViewerStore) {
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
let decryptFailTimestamps = [];
let isHealingInProgress = false;

function trackDecryptionFailure() {
  const now = Date.now();
  decryptFailTimestamps.push(now);
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
    decryptFailTimestamps = [];

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
// 🛠️ UTILITY FUNCTIONS & MIME CHECKS
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

// ======================================================================
// 🔐 MONGODB AUTH & SESSION MANAGEMENT
// ======================================================================

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
    clearAll,
    clearSessionKeys
  };
}

// ======================================================================
// 📥 PERSISTENT USER MEDIA BUFFER MANAGEMENT
// ======================================================================

async function addToUserBuffer(chatId, senderId, senderName, mediaItem, msgId) {
  const model = getPendingMediaModel();
  if (model) {
    try {
      await model.create({
        chatId,
        senderId,
        senderName,
        messageId: msgId || crypto.randomBytes(8).toString('hex'),
        type: mediaItem.type,
        data: mediaItem.data,
        mimeType: mediaItem.mimeType,
        caption: mediaItem.caption || '',
        fileName: mediaItem.fileName || ''
      });
      const count = await model.countDocuments({ chatId, senderId, processed: false });
      log('💾', `Saved media [${mediaItem.type}] to MongoDB buffer (Total: ${count})`);
      return count;
    } catch (err) {
      log('⚠️', `MongoDB buffer save error: ${err.message}. Falling back to RAM.`);
    }
  }

  // RAM Fallback
  if (!ramMediaBuffers.has(chatId)) ramMediaBuffers.set(chatId, new Map());
  const chatBuf = ramMediaBuffers.get(chatId);
  if (!chatBuf.has(senderId)) chatBuf.set(senderId, []);
  const buf = chatBuf.get(senderId);
  buf.push(mediaItem);
  return buf.length;
}

async function clearUserBuffer(chatId, senderId) {
  const model = getPendingMediaModel();
  let mongoItems = [];

  if (model) {
    try {
      const docs = await model.find({ chatId, senderId, processed: false }).sort({ createdAt: 1 });
      if (docs.length > 0) {
        await model.updateMany({ chatId, senderId, processed: false }, { processed: true });
        mongoItems = docs.map(d => ({
          type: d.type,
          data: d.data,
          mimeType: d.mimeType,
          caption: d.caption,
          fileName: d.fileName
        }));
      }
    } catch (err) {
      log('⚠️', `MongoDB clear buffer error: ${err.message}`);
    }
  }

  // Also check RAM buffer fallback
  let ramItems = [];
  if (ramMediaBuffers.has(chatId)) {
    const chatBuf = ramMediaBuffers.get(chatId);
    if (chatBuf.has(senderId)) {
      ramItems = chatBuf.get(senderId);
      chatBuf.delete(senderId);
    }
  }

  return [...mongoItems, ...ramItems];
}

async function getUserBufferCount(chatId, senderId) {
  const model = getPendingMediaModel();
  let count = 0;
  if (model) {
    try {
      count = await model.countDocuments({ chatId, senderId, processed: false });
    } catch (err) {}
  }
  if (ramMediaBuffers.has(chatId) && ramMediaBuffers.get(chatId).has(senderId)) {
    count += ramMediaBuffers.get(chatId).get(senderId).length;
  }
  return count;
}

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
  if (processedMessageIds.has(msgId)) return true;
  processedMessageIds.add(msgId);
  if (processedMessageIds.size > MAX_PROCESSED_IDS) {
    const arr = Array.from(processedMessageIds);
    arr.slice(0, Math.floor(MAX_PROCESSED_IDS / 2)).forEach(id => processedMessageIds.delete(id));
  }
  return false;
}

// Track pending empty messages
const pendingEmptyMessages = new Map();

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
    return [mediaFiles];
  }

  const batches = [];
  let currentBatch = [];
  let activeCaption = null;

  for (const file of mediaFiles) {
    const fileCaption = (file.caption || '').trim();

    if (fileCaption && fileCaption !== activeCaption) {
      if (currentBatch.length > 0) {
        batches.push(currentBatch);
      }
      currentBatch = [file];
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
// 🔄 TIMEOUT LOGIC
// ======================================================================
function resetUserTimeout(chatId, senderId, senderName) {
  if (!chatTimeouts.has(chatId)) {
    chatTimeouts.set(chatId, new Map());
  }
  const chatTimeoutMap = chatTimeouts.get(chatId);

  if (chatTimeoutMap.has(senderId)) {
    clearTimeout(chatTimeoutMap.get(senderId));
  }

  const isCTGroup = chatId === CONFIG.GROUPS.CT_SOURCE || chatId === CONFIG.GROUPS.CT_TARGET;
  const isMRIGroup = chatId === CONFIG.GROUPS.MRI_SOURCE || chatId === CONFIG.GROUPS.MRI_TARGET;
  const isAutoGroup = isCTGroup || isMRIGroup;

  const delay = isAutoGroup ? CONFIG.AUTO_PROCESS_DELAY_MS : CONFIG.MEDIA_TIMEOUT_MS;
  const shortId = getShortSenderId(senderId);

  const timeoutCallback = async () => {
    if (isAutoGroup) {
      const mediaFiles = await clearUserBuffer(chatId, senderId);
      if (mediaFiles.length > 0) {
        log('⏱️', `Auto-processing ${mediaFiles.length} item(s) from Auto Group (${isCTGroup ? 'CT' : 'MRI'})`);

        const targetChatId = isCTGroup ? CONFIG.GROUPS.CT_TARGET : CONFIG.GROUPS.MRI_TARGET;

        if (targetChatId) {
          const batches = groupMediaSmartly(mediaFiles);

          if (batches.length > 1) {
            log('🔀', `Detected ${batches.length} distinct patient contexts. Processing separately.`);
          }

          for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            if (batch.length === 0) continue;

            log('▶️', `Processing Batch ${i+1}/${batches.length} (${batch.length} files)`);
            await processMedia(sock, chatId, batch, false, null, senderId, senderName, null, 3, false, targetChatId);

            if (i < batches.length - 1) {
              await new Promise(r => setTimeout(r, 2000));
            }
          }

        } else {
          log('⚠️', 'Target group not configured for this source!');
        }
      }
    } else {
      const clearedItems = await clearUserBuffer(chatId, senderId);
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

// ======================================================================
// 🎥 SMART VIDEO FRAME EXTRACTION
// ======================================================================
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

// Express App Setup
const app = express();
const PORT = process.env.PORT || 3000;

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
            <div class="icon">⌛</div>
            <h1>Link Expired or Invalid</h1>
            <p>This media viewer link has expired (12 hour limit) or does not exist.</p>
        </div>
    </body>
    </html>`);
  }

  if (Date.now() >= entry.expiresAt) {
    mediaViewerStore.delete(viewerId);
    return res.status(410).send('Link Expired');
  }

  let mediaHtml = '';
  entry.media.forEach((item, index) => {
    if (item.type === 'image') {
      mediaHtml += `<div class="media-card"><img src="data:${item.mimeType};base64,${item.data}" alt="Medical Image ${index+1}"/>${item.caption ? `<p class="caption">${item.caption}</p>` : ''}</div>`;
    } else if (item.type === 'pdf') {
      mediaHtml += `<div class="media-card"><object data="data:application/pdf;base64,${item.data}" type="application/pdf" width="100%" height="600px"><p>PDF document: ${item.fileName || 'Report.pdf'}</p></object></div>`;
    } else if (item.type === 'audio' || item.type === 'voice') {
      mediaHtml += `<div class="media-card"><audio controls src="data:${item.mimeType};base64,${item.data}"></audio></div>`;
    } else if (item.type === 'video') {
      mediaHtml += `<div class="media-card"><video controls width="100%" src="data:${item.mimeType};base64,${item.data}"></video></div>`;
    }
  });

  res.send(`
  <!DOCTYPE html>
  <html>
  <head>
      <title>Medical Media Viewer</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; padding: 20px; }
          .header { text-align: center; margin-bottom: 30px; padding-bottom: 15px; border-bottom: 1px solid #1e293b; }
          .header h1 { margin: 0; color: #38bdf8; font-size: 24px; }
          .grid { display: flex; flex-direction: column; gap: 20px; max-width: 900px; margin: 0 auto; }
          .media-card { background: #1e293b; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
          .media-card img { width: 100%; height: auto; border-radius: 8px; display: block; }
          .caption { margin-top: 10px; color: #cbd5e1; font-size: 14px; }
      </style>
  </head>
  <body>
      <div class="header">
          <h1>📋 Medical Reports & Media (${entry.media.length} items)</h1>
      </div>
      <div class="grid">${mediaHtml}</div>
  </body>
  </html>`);
});

app.get('/health', (req, res) => res.status(200).send('OK'));

app.listen(PORT, () => log('🌐', `HTTP Server listening on port ${PORT}`));

// ======================================================================
// 🤖 GEMINI API & AI INFERENCE LOGIC
// ======================================================================

async function callGeminiApi(modelName, apiKey, promptParts, systemInstruction) {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: systemInstruction
  });

  const safetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE }
  ];

  const result = await model.generateContent({
    contents: [{ role: 'user', parts: promptParts }],
    safetySettings
  });

  return result.response.text();
}

async function runGeminiWithFallback(promptParts, systemInstruction) {
  const models = [
    CONFIG.GEMINI_MODEL,
    CONFIG.GEMINI_MODEL_FALLBACK_1,
    CONFIG.GEMINI_MODEL_FALLBACK_2,
    CONFIG.GEMINI_MODEL_FALLBACK_3
  ];

  const apiKeys = CONFIG.API_KEYS;
  if (!apiKeys || apiKeys.length === 0) {
    throw new Error('No GEMINI_API_KEYS configured');
  }

  let lastErr = null;

  for (const apiKey of apiKeys) {
    for (const modelName of models) {
      try {
        log('🧠', `Calling Gemini API (${modelName}) with key ...${apiKey.slice(-4)}`);
        const text = await callGeminiApi(modelName, apiKey, promptParts, systemInstruction);
        if (text && text.trim().length > 0) return text;
      } catch (err) {
        log('⚠️', `Model ${modelName} failed with key ...${apiKey.slice(-4)}: ${err.message}`);
        lastErr = err;
      }
    }
  }

  throw lastErr || new Error('All Gemini models and API keys failed.');
}

// ======================================================================
// 🔄 MEDIA PROCESSOR & SENDER
// ======================================================================

async function processMedia(sock, chatId, mediaFiles, isFollowUp = false, followUpText = null, senderId = null, senderName = null, originalMsgKey = null, retriesLeft = 3, isSecondary = false, customTargetChatId = null) {
  try {
    const promptParts = [];

    for (const file of mediaFiles) {
      if (file.type === 'text') {
        promptParts.push({ text: `Accompanying Note/Text: ${file.caption}` });
      } else if (file.type === 'video') {
        try {
          const videoBuf = Buffer.from(file.data, 'base64');
          const frames = await extractFramesFromVideo(videoBuf, 3);
          frames.forEach(frameB64 => {
            promptParts.push({
              inlineData: { data: frameB64, mimeType: 'image/jpeg' }
            });
          });
        } catch (e) {
          log('⚠️', `Video frame extraction failed: ${e.message}`);
        }
      } else {
        promptParts.push({
          inlineData: { data: file.data, mimeType: file.mimeType }
        });
      }
    }

    if (isFollowUp && followUpText) {
      promptParts.push({ text: `User Follow-Up Query: ${followUpText}` });
    }

    const sysInst = isSecondary ? SECONDARY_SYSTEM_INSTRUCTION : CONFIG.SYSTEM_INSTRUCTION;
    const responseText = await runGeminiWithFallback(promptParts, sysInst);

    const targetChat = customTargetChatId || chatId;

    let viewerUrl = null;
    const viewerId = storeMediaForViewer(mediaFiles);
    if (viewerId) {
      const baseUrl = process.env.RENDER_EXTERNAL_URL || `http://localhost:${PORT}`;
      viewerUrl = `${baseUrl}/view/${viewerId}`;
    }

    let finalMessage = responseText;
    if (viewerUrl) {
      finalMessage += `\n\n🔗 View Original Files: ${viewerUrl}`;
    }

    const sentMsg = await sock.sendMessage(targetChat, { text: finalMessage });
    if (sentMsg && sentMsg.key) {
      trackBotMessage(targetChat, sentMsg.key.id);
      storeContext(targetChat, sentMsg.key.id, mediaFiles, responseText, senderId);
    }

    log('✅', `Successfully processed & sent response to ${targetChat}`);

  } catch (err) {
    log('❌', `Process media error: ${err.message}`);
    if (retriesLeft > 0) {
      log('🔄', `Retrying media processing (${retriesLeft} retries remaining)...`);
      await new Promise(r => setTimeout(r, 3000));
      return processMedia(sock, chatId, mediaFiles, isFollowUp, followUpText, senderId, senderName, originalMsgKey, retriesLeft - 1, isSecondary, customTargetChatId);
    }
  }
}

// ======================================================================
// 📥 PERSISTENT DOWNLOAD WITH EXPONENTIAL BACKOFF (SURVIVES CDN DROPS)
// ======================================================================

async function downloadMediaWithRetry(msg, maxRetries = 5) {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      const buffer = await downloadMediaMessage(
        msg,
        'buffer',
        {},
        { logger: pino({ level: 'silent' }), reuploadRequest: sock?.updateMediaMessage }
      );
      if (buffer && buffer.length > 0) return buffer;
    } catch (err) {
      attempt++;
      log('⚠️', `Media download retry ${attempt}/${maxRetries}: ${err.message}`);
      if (attempt >= maxRetries) throw err;
      await new Promise(res => setTimeout(res, 1500 * Math.pow(2, attempt)));
    }
  }
  return null;
}

// ======================================================================
// 🚀 MAIN BAILEYS SOCKET INITIALIZATION & STARTUP RECOVERY
// ======================================================================

async function startBot() {
  try {
    const baileysModule = await import('@whiskeysockets/baileys');
    makeWASocket = baileysModule.default;
    DisconnectReason = baileysModule.DisconnectReason;
    downloadMediaMessage = baileysModule.downloadMediaMessage;
    fetchLatestBaileysVersion = baileysModule.fetchLatestBaileysVersion;

    if (CONFIG.MONGODB_URI) {
      try {
        await mongoose.connect(CONFIG.MONGODB_URI);
        mongoConnected = true;
        log('🟢', 'Connected to MongoDB');

        // Startup Recovery: Check for un-processed pending media from previous server restarts
        const model = getPendingMediaModel();
        if (model) {
          const pendingChats = await model.distinct('chatId', { processed: false });
          if (pendingChats.length > 0) {
            log('🔄', `Startup Recovery: Found pending un-processed media in ${pendingChats.length} chat(s)`);
            for (const cId of pendingChats) {
              const senders = await model.distinct('senderId', { chatId: cId, processed: false });
              for (const sId of senders) {
                resetUserTimeout(cId, sId, 'Restored User');
              }
            }
          }
        }

      } catch (mErr) {
        log('❌', `MongoDB connection failed: ${mErr.message}`);
      }
    }

    authState = await useMongoDBAuthState();
    const { version } = await fetchLatestBaileysVersion();

    sock = makeWASocket({
      version,
      logger: pino({ level: 'silent' }),
      printQRInTerminal: true,
      auth: authState.state,
      browser: ['WhatsApp Medical Bot', 'Chrome', '1.0.0']
    });

    sock.ev.on('creds.update', authState.saveCreds);

    sock.ev.on('connection.update', (update) => {
      const { connection, lastDisconnect, qr } = update;

      if (qr) {
        QRCode.toDataURL(qr, (err, url) => {
          if (!err) qrCodeDataURL = url;
        });
      }

      if (connection === 'open') {
        isConnected = true;
        qrCodeDataURL = null;
        botStatus = 'Connected & Listening 🟢';
        log('🚀', 'WhatsApp Bot is Connected!');
      }

      if (connection === 'close') {
        isConnected = false;
        const reason = lastDisconnect?.error?.output?.statusCode || lastDisconnect?.error?.message;
        log('⚠️', `Connection closed. Reason: ${reason}`);

        if (reason !== DisconnectReason.loggedOut) {
          setTimeout(startBot, 5000);
        } else {
          log('❌', 'Logged out. Clearing auth...');
          authState.clearAll();
          setTimeout(startBot, 5000);
        }
      }
    });

    sock.ev.on('messages.upsert', async (m) => {
      if (!m.messages || m.messages.length === 0) return;
      const msg = m.messages[0];
      if (msg.key.fromMe) return;

      const msgId = msg.key.id;
      const chatId = msg.key.remoteJid;
      const senderId = getSenderId(msg);
      const senderName = getSenderName(msg);

      if (isMessageAlreadyProcessed(msgId)) return;

      const messageType = Object.keys(msg.message || {})[0];
      if (!messageType) return;

      // Handle Decryption Failure Logging
      if (messageType === 'ciphertext' || messageType === 'protocolMessage') {
        trackDecryptionFailure();
        return;
      }

      const isImage = messageType === 'imageMessage';
      const isDocument = messageType === 'documentMessage';
      const isAudio = messageType === 'audioMessage';
      const isVideo = messageType === 'videoMessage';
      const isText = messageType === 'conversation' || messageType === 'extendedTextMessage';

      if (isImage || isDocument || isAudio || isVideo) {
        try {
          const buffer = await downloadMediaWithRetry(msg, 5);
          if (!buffer) return;

          const mediaContent = msg.message.imageMessage || msg.message.documentMessage || msg.message.audioMessage || msg.message.videoMessage;
          const mimeType = mediaContent.mimetype || 'image/jpeg';
          const caption = mediaContent.caption || '';
          const fileName = mediaContent.fileName || '';

          const mediaItem = {
            type: isImage ? 'image' : (isDocument ? 'pdf' : (isAudio ? 'audio' : 'video')),
            data: buffer.toString('base64'),
            mimeType: mimeType,
            caption: caption,
            fileName: fileName
          };

          await addToUserBuffer(chatId, senderId, senderName, mediaItem, msgId);
          resetUserTimeout(chatId, senderId, senderName);

        } catch (err) {
          log('❌', `Error ingesting incoming media: ${err.message}`);
        }
      } else if (isText) {
        const text = msg.message.conversation || msg.message.extendedTextMessage?.text || '';
        const contextInfo = msg.message.extendedTextMessage?.contextInfo;
        const quotedMsgId = contextInfo?.stanzaId;

        if (quotedMsgId && getStoredContext(chatId, quotedMsgId)) {
          const ctx = getStoredContext(chatId, quotedMsgId);
          log('💬', `Handling follow-up query for quote ${quotedMsgId.substring(0, 8)}...`);
          await processMedia(sock, chatId, ctx.mediaFiles, true, text, senderId, senderName, quotedMsgKey);
        } else if (COMMANDS.includes(text.toLowerCase().trim())) {
          // Handle commands (.1, .2, etc.)
          if (text === '.' || text === 'status') {
            const count = await getUserBufferCount(chatId, senderId);
            await sock.sendMessage(chatId, { text: `📊 Status: ${count} media file(s) currently buffered for you.` });
          } else if (text === 'clear') {
            await clearUserBuffer(chatId, senderId);
            clearUserTimeout(chatId, senderId);
            await sock.sendMessage(chatId, { text: '🧹 Cleared your buffered media.' });
          }
        }
      }
    });

  } catch (err) {
    log('❌', `Start bot error: ${err.message}`);
    setTimeout(startBot, 10000);
  }
}

startBot();
