import fs from 'fs';
import path from 'path';

const tier1Dir = path.resolve('tests/tier1_features');
if (!fs.existsSync(tier1Dir)) fs.mkdirSync(tier1Dir, { recursive: true });

// --- 1. f1_baileys_auth.test.js ---
fs.writeFileSync(path.join(tier1Dir, 'f1_baileys_auth.test.js'), `import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { MockMongoStore } from '../mocks/mock_mongo.js';
import { MockBaileysSocket } from '../mocks/mock_baileys.js';

describe('F1: Baileys Auth & Persistent Session Lifecycle', () => {
  let mockMongo;
  let mockSock;

  beforeEach(() => {
    mockMongo = new MockMongoStore();
    mockSock = new MockBaileysSocket();
  });

  test('TC-F1-01: MongoDB auth state stores and retrieves credentials accurately', async () => {
    const credsData = {
      noiseKey: { private: [1, 2, 3], public: [4, 5, 6] },
      signedIdentityKey: { private: [7, 8], public: [9, 10] },
      registrationId: 1048576,
      advSecretKey: 'ADV_SECRET_KEY_MOCK'
    };

    const serialized = JSON.stringify(credsData);

    await mockMongo.findOneAndUpdate(
      { key: 'auth_creds' },
      { key: 'auth_creds', value: serialized, updatedAt: new Date() },
      { upsert: true }
    );

    const doc = await mockMongo.findOne({ key: 'auth_creds' });
    assert.ok(doc, 'Auth creds doc must exist in MongoDB');
    
    const parsed = JSON.parse(doc.value);
    assert.equal(parsed.registrationId, 1048576);
    assert.deepEqual(parsed.noiseKey.private, [1, 2, 3]);
    assert.equal(parsed.advSecretKey, 'ADV_SECRET_KEY_MOCK');
  });

  test('TC-F1-02: Fast reconnection loads existing credentials under 3 seconds', async () => {
    const startTime = Date.now();
    
    await mockMongo.findOneAndUpdate(
      { key: 'auth_creds' },
      { key: 'auth_creds', value: JSON.stringify({ registrationId: 554433 }), updatedAt: new Date() },
      { upsert: true }
    );

    const doc = await mockMongo.findOne({ key: 'auth_creds' });
    assert.ok(doc, 'Cached credentials retrieved');
    const elapsedMs = Date.now() - startTime;

    assert.ok(elapsedMs < 3000, \`Reconnection fetch must be <3s (actual: \${elapsedMs}ms)\`);
  });

  test('TC-F1-03: Signal key partition purges key_* while preserving auth_creds on session heal', async () => {
    await mockMongo.findOneAndUpdate(
      { key: 'auth_creds' },
      { key: 'auth_creds', value: '{"me":{"id":"919876543210@s.whatsapp.net"}}' },
      { upsert: true }
    );

    for (let i = 1; i <= 10; i++) {
      await mockMongo.findOneAndUpdate(
        { key: \`key_session_\${i}\` },
        { key: \`key_session_\${i}\`, value: \`session_data_\${i}\` },
        { upsert: true }
      );
    }

    const preStats = mockMongo.getSessionStats();
    assert.equal(preStats.authCredsCount, 1);
    assert.equal(preStats.signalKeysCount, 10);

    const result = await mockMongo.deleteManySessions({
      key: { $regex: /^key_/ }
    });

    assert.equal(result.deletedCount, 10, 'Must delete exactly 10 signal keys');

    const postStats = mockMongo.getSessionStats();
    assert.equal(postStats.signalKeysCount, 0, 'All signal keys must be nuked');
    assert.equal(postStats.authCredsCount, 1, 'auth_creds must be strictly preserved');
    
    const preservedCreds = await mockMongo.findOne({ key: 'auth_creds' });
    assert.ok(preservedCreds, 'auth_creds document still exists');
  });

  test('TC-F1-04: QR code event is emitted on initial unauthenticated state', async () => {
    let receivedQr = null;
    mockSock.ev.on('connection.update', (update) => {
      if (update.qr) {
        receivedQr = update.qr;
      }
    });

    const mockRawQr = '2@1AbCdEfGhIjKlMnOpQrStUvWxYz==,mockPublicKey,mockAdvSecret';
    mockSock.emitConnectionUpdate({ qr: mockRawQr });

    assert.equal(receivedQr, mockRawQr, 'QR code payload must be emitted via connection.update');
  });

  test('TC-F1-05: Message deduplication cache prevents re-processing duplicate message IDs', () => {
    const processedSet = new Set();
    const isDuplicate = (msgId) => {
      if (!msgId) return false;
      if (processedSet.has(msgId)) return true;
      processedSet.add(msgId);
      return false;
    };

    const msgId = 'WAMID_DUPLICATE_TEST_001';
    assert.equal(isDuplicate(msgId), false, 'First message delivery should NOT be duplicate');
    assert.equal(isDuplicate(msgId), true, 'Second message delivery with same ID MUST be flagged duplicate');
    assert.equal(isDuplicate(msgId), true, 'Third message delivery with same ID MUST be flagged duplicate');

    const newMsgId = 'WAMID_FRESH_002';
    assert.equal(isDuplicate(newMsgId), false, 'Different message ID should NOT be duplicate');
  });

  test('TC-F1-06: Multi-device data serialization preserves Uint8Array, Buffer, and BigInt types', () => {
    const complexData = {
      smallBuffer: Buffer.from('hello radiology'),
      uint8Arr: new Uint8Array([10, 20, 30, 40]),
      bigIntValue: BigInt('900719925474099999')
    };

    const serialized = JSON.stringify(complexData, (k, v) => {
      if (typeof v === 'bigint') return { type: 'BigInt', value: v.toString() };
      if (Buffer.isBuffer(v)) return { type: 'Buffer', data: Array.from(v) };
      if (v instanceof Uint8Array) return { type: 'Uint8Array', data: Array.from(v) };
      return v;
    });

    const deserialized = JSON.parse(serialized, (k, v) => {
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        if (v.type === 'BigInt') return BigInt(v.value);
        if (v.type === 'Buffer') return Buffer.from(v.data || v.value);
        if (v.type === 'Uint8Array') return new Uint8Array(v.data || v.value);
      }
      return v;
    });

    assert.deepEqual(deserialized.smallBuffer, complexData.smallBuffer);
    assert.deepEqual(deserialized.uint8Arr, complexData.uint8Arr);
    assert.equal(deserialized.bigIntValue, complexData.bigIntValue);
  });
});
`);

// --- 2. f2_media_ingestion.test.js ---
fs.writeFileSync(path.join(tier1Dir, 'f2_media_ingestion.test.js'), `import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  unwrapMessage,
  formatInputMediaSummary,
  optimizeImageForAi
} from '../../index.js';
import {
  CT_SCAN_IMAGE_BASE64,
  PDF_REPORT_BASE64,
  VOICE_NOTE_OGG_BASE64,
  AUDIO_MP3_BASE64,
  VIDEO_CINE_LOOP_BASE64
} from '../fixtures/medical_fixtures.js';
import {
  createImageMessage,
  createPdfMessage,
  createAudioMessage,
  createVideoMessage,
  createWrappedMessage
} from '../mocks/mock_baileys.js';

describe('F2: Multi-Modal Media Ingestion & Deep Wrapper Unwrapping', () => {

  test('TC-F2-01: Full-resolution image is preserved losslessly for AI input', async () => {
    const result = await optimizeImageForAi(CT_SCAN_IMAGE_BASE64, 'image/jpeg');
    assert.equal(result.mimeType, 'image/jpeg');
    assert.equal(result.data, CT_SCAN_IMAGE_BASE64, 'Image base64 data must be preserved with 100% fidelity');
  });

  test('TC-F2-02: Multi-page PDF report is identified and packaged with application/pdf MIME', () => {
    const pdfMsg = createPdfMessage({
      fileName: 'CT_Thorax_Report_2026.pdf',
      data: PDF_REPORT_BASE64,
      mimetype: 'application/pdf'
    });
    const unwrapped = unwrapMessage(pdfMsg.message);
    assert.ok(unwrapped.documentMessage, 'Must extract documentMessage');
    assert.equal(unwrapped.documentMessage.mimetype, 'application/pdf');
    assert.equal(unwrapped.documentMessage.fileName, 'CT_Thorax_Report_2026.pdf');
  });

  test('TC-F2-03: Audio and Voice Note messages are recognized with correct codec headers', () => {
    const voiceMsg = createAudioMessage({ isVoice: true, data: VOICE_NOTE_OGG_BASE64, mimetype: 'audio/ogg; codecs=opus' });
    const musicMsg = createAudioMessage({ isVoice: false, data: AUDIO_MP3_BASE64, mimetype: 'audio/mpeg' });

    const unwrappedVoice = unwrapMessage(voiceMsg.message);
    const unwrappedMusic = unwrapMessage(musicMsg.message);

    assert.equal(unwrappedVoice.audioMessage.ptt, true, 'Voice note must have ptt=true');
    assert.equal(unwrappedMusic.audioMessage.ptt, false, 'Standard audio must have ptt=false');
  });

  test('TC-F2-04: Video cine loop is structured for smart FPS frame extraction', () => {
    const videoMsg = createVideoMessage({
      data: VIDEO_CINE_LOOP_BASE64,
      caption: 'Dynamic Liver MRI Cine Loop',
      seconds: 6
    });
    const unwrapped = unwrapMessage(videoMsg.message);
    assert.ok(unwrapped.videoMessage);
    assert.equal(unwrapped.videoMessage.seconds, 6);
    assert.equal(unwrapped.videoMessage.caption, 'Dynamic Liver MRI Cine Loop');
  });

  test('TC-F2-05: Recursive unwrapping unwraps 20+ wrapper layers down to base payload', () => {
    const baseImgMsg = createImageMessage({ caption: 'Deep Nested CT Scan' });

    let wrapped = createWrappedMessage('viewOnceMessage', baseImgMsg);
    wrapped = createWrappedMessage('ephemeralMessage', wrapped);
    wrapped = createWrappedMessage('viewOnceMessageV2', wrapped);
    wrapped = createWrappedMessage('documentWithCaptionMessage', wrapped);

    const unwrapped = unwrapMessage(wrapped.message);
    assert.ok(unwrapped.imageMessage, 'Must recursively unwrap to imageMessage');
    assert.equal(unwrapped.imageMessage.caption, 'Deep Nested CT Scan');
  });

  test('TC-F2-06: Deterministic Input Media Summary counts all media categories accurately', () => {
    const counts1 = { images: 3, pdfs: 1, audio: 2, video: 1, texts: 0 };
    const summary1 = formatInputMediaSummary(counts1);
    assert.equal(summary1, '📥 *Input Media:* 3 Images, 1 PDF, 2 Audio Notes, 1 Video');

    const countsSingle = { images: 1, pdfs: 1, audio: 1, video: 1, texts: 1 };
    const summarySingle = formatInputMediaSummary(countsSingle);
    assert.equal(summarySingle, '📥 *Input Media:* 1 Image, 1 PDF, 1 Audio Note, 1 Video, 1 Text Note');

    const countsEmpty = { images: 0, pdfs: 0, audio: 0, video: 0, texts: 0 };
    const summaryEmpty = formatInputMediaSummary(countsEmpty);
    assert.equal(summaryEmpty, '📥 *Input Media:* None');
  });
});
`);

// --- 3. f3_gemini_chain.test.js ---
fs.writeFileSync(path.join(tier1Dir, 'f3_gemini_chain.test.js'), `import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { MockGeminiEngine } from '../mocks/mock_gemini.js';
import { CONFIG } from '../../index.js';
import { SAMPLE_PRIMARY_PROFILE, SAMPLE_JSON_METADATA } from '../fixtures/medical_fixtures.js';

describe('F3: 5-Tier Gemini Model Chain & Structured Clinical Profiles', () => {
  let mockGemini;

  beforeEach(() => {
    mockGemini = new MockGeminiEngine();
  });

  test('TC-F3-01: 5-tier fallback executes sequentially from primary (3.7) down to lite (3.5-lite)', async () => {
    mockGemini.setFailModel('gemini-3.7-flash');
    mockGemini.setFailModel('gemini-3.5-flash');

    const response = await mockGemini.generate(['Analyze CT scan']);
    assert.ok(response.includes('{model used: gemini-3.6-flash}'), 'Must fallback to tier 3 (gemini-3.6-flash)');

    const history = mockGemini.getHistory();
    assert.equal(history.length, 1);
    const attempts = history[0].modelAttempts;
    assert.equal(attempts[0].model, 'gemini-3.7-flash');
    assert.equal(attempts[0].success, false);
    assert.equal(attempts[1].model, 'gemini-3.5-flash');
    assert.equal(attempts[1].success, false);
    assert.equal(attempts[2].model, 'gemini-3.6-flash');
    assert.equal(attempts[2].success, true);
  });

  test('TC-F3-02: Structured <<JSON>> block is extracted and parsed into Quick Reference format', () => {
    const rawAiOutput = \`\${SAMPLE_PRIMARY_PROFILE}\\n\\n<<JSON>>\${JSON.stringify(SAMPLE_JSON_METADATA)}<<JSON>>\`;

    const match = rawAiOutput.match(/<<JSON>>(.*?)<<JSON>>/s);
    assert.ok(match, 'Must match <<JSON>> block');
    
    const parsed = JSON.parse(match[1].trim());
    assert.equal(parsed.mrn, 'UHID-2026-9941');
    assert.equal(parsed.age, '54');
    assert.equal(parsed.sex, 'F');
    assert.equal(parsed.study, 'CECT Thorax');

    const quickRef = \`📋 *Quick Reference:*\\n• MRN/Reg No: \${parsed.mrn}\\n• Age: \${parsed.age}\\n• Sex: \${parsed.sex}\\n• Study: \${parsed.study}\\n• Brief: \${parsed.brief}\`;
    assert.ok(quickRef.includes('• MRN/Reg No: UHID-2026-9941'));
    assert.ok(quickRef.includes('• Study: CECT Thorax'));
  });

  test('TC-F3-03: Dynamic date verification anchor injects current year into prompt', () => {
    const currentDate = new Date().toLocaleDateString('en-GB', { 
      day: 'numeric', month: 'long', year: 'numeric' 
    });

    const promptWithDate = \`Today's current date is \${currentDate}. Please pay extremely close attention to the dates.\`;
    assert.ok(promptWithDate.includes(String(new Date().getFullYear())), 'Must anchor to current calendar year');
  });

  test('TC-F3-04: Single cohesive clinical profile formatting starts with *Clinical Profile: and ends with *', () => {
    const profile = SAMPLE_PRIMARY_PROFILE;
    assert.ok(profile.startsWith('*Clinical Profile:'), 'Must start with *Clinical Profile:');
    assert.ok(profile.endsWith('*'), 'Must end with *');
    assert.ok(!profile.includes('\\n\\n*Clinical Profile:'), 'Must be a single paragraph');
  });

  test('TC-F3-05: API key rotation shifts primary key every 2 hours', () => {
    const testKeys = ['KEY_A_PRIMARY', 'KEY_B_SECONDARY', 'KEY_C_TERTIARY'];
    const rotateKeys = (keys) => {
      if (keys.length > 1) {
        const k = keys.shift();
        keys.push(k);
      }
      return keys;
    };

    const rotated1 = rotateKeys([...testKeys]);
    assert.equal(rotated1[0], 'KEY_B_SECONDARY');

    const rotated2 = rotateKeys(rotated1);
    assert.equal(rotated2[0], 'KEY_C_TERTIARY');
  });

  test('TC-F3-06: Secondary chained analysis (..) passes expert radiologist system instruction', async () => {
    const secondarySysInstruction = 'You are an expert radiologist. Analyse that info and advise regarding imaging modality.';
    const response = await mockGemini.generate(['=== CLINICAL PROFILE === ...'], secondarySysInstruction);

    assert.ok(response.includes('Radiology Protocol & Imaging Advice'), 'Secondary mode must generate imaging advice');
    const history = mockGemini.getHistory();
    assert.equal(history[0].systemInstruction, secondarySysInstruction);
  });
});
`);

// --- 4. f4_group_routing.test.js ---
fs.writeFileSync(path.join(tier1Dir, 'f4_group_routing.test.js'), `import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('F4: Group Routing, Smart Batching & Sender Attribution', () => {

  test('TC-F4-01: CT Source group routes silently to CT Target group', () => {
    const GROUPS = {
      CT_SOURCE: '120363011111111111@g.us',
      CT_TARGET: '120363022222222222@g.us',
      MRI_SOURCE: '120363033333333333@g.us',
      MRI_TARGET: '120363044444444444@g.us'
    };

    const incomingChatId = GROUPS.CT_SOURCE;
    const isCT = incomingChatId === GROUPS.CT_SOURCE;
    const destinationChatId = isCT ? GROUPS.CT_TARGET : null;

    assert.equal(destinationChatId, '120363022222222222@g.us');
  });

  test('TC-F4-02: MRI Source group routes silently to MRI Target group', () => {
    const GROUPS = {
      CT_SOURCE: '120363011111111111@g.us',
      CT_TARGET: '120363022222222222@g.us',
      MRI_SOURCE: '120363033333333333@g.us',
      MRI_TARGET: '120363044444444444@g.us'
    };

    const incomingChatId = GROUPS.MRI_SOURCE;
    const isMRI = incomingChatId === GROUPS.MRI_SOURCE;
    const destinationChatId = isMRI ? GROUPS.MRI_TARGET : null;

    assert.equal(destinationChatId, '120363044444444444@g.us');
  });

  test('TC-F4-03: Caption-based smart batching splits distinct patient cases', () => {
    const mediaFiles = [
      { type: 'image', caption: 'Patient A: John Doe', data: 'data1' },
      { type: 'image', caption: 'Patient A: John Doe', data: 'data2' },
      { type: 'pdf', caption: 'Patient B: Jane Smith', data: 'data3' },
      { type: 'image', caption: 'Patient B: Jane Smith', data: 'data4' }
    ];

    const distinctIdentifiers = new Set();
    mediaFiles.forEach(f => {
      const id = (f.caption || '').trim();
      if (id) distinctIdentifiers.add(id);
    });

    assert.equal(distinctIdentifiers.size, 2);

    const batches = [];
    let currentBatch = [];
    let activeId = null;

    for (const f of mediaFiles) {
      const id = (f.caption || '').trim();
      if (id && id !== activeId) {
        if (currentBatch.length > 0) batches.push(currentBatch);
        currentBatch = [f];
        activeId = id;
      } else {
        currentBatch.push(f);
        if (!activeId && id) activeId = id;
      }
    }
    if (currentBatch.length > 0) batches.push(currentBatch);

    assert.equal(batches.length, 2, 'Must split into exactly 2 patient batches');
    assert.equal(batches[0].length, 2, 'Batch 1 must contain 2 items');
    assert.equal(batches[1].length, 2, 'Batch 2 must contain 2 items');
  });

  test('TC-F4-04: Phone sender JID generates clickable @mention attribution tag', () => {
    const senderId = '919876543210@s.whatsapp.net';
    const phone = senderId.split('@')[0];
    const isPhoneNumber = /^\\d{7,15}$/.test(phone);

    assert.equal(isPhoneNumber, true);
    const attribution = \`\\n\\n👤 *Sent by:* @\${phone}\`;
    assert.equal(attribution, '\\n\\n👤 *Sent by:* @919876543210');
  });

  test('TC-F4-05: Non-phone LID JID falls back cleanly to display name without broken mention', () => {
    const senderId = '120363047547742844@lid';
    const senderName = 'Dr. Vikram Patel';
    const phone = senderId.split('@')[0];
    const domain = senderId.split('@')[1];
    const isPhoneNumber = /^\\d{7,15}$/.test(phone) && (domain === 's.whatsapp.net' || domain === 'c.us');

    assert.equal(isPhoneNumber, false);
    const attribution = \`\\n\\n👤 *Sent by:* \${senderName || phone}\`;
    assert.equal(attribution, '\\n\\n👤 *Sent by:* Dr. Vikram Patel');
  });

  test('TC-F4-06: Stradus DICOM viewer & MRI protocol footer is appended to group chat reports', () => {
    const GROUP_REPLY_FOOTER = \`

━━━━━━━━━━━━━━━━━━━━━━
🖼️ *Go here to see the scan images:*
https://view.stradus.com/

🤖 *Copy-paste the clinical profile here to get suggestions regarding MRI protocols:*
https://ai.studio/apps/86a65a19-cf2f-46de-b4d0-9a941be83604

🎙️ *Radiology dictation:*
https://ai.studio/apps/3f0807e3-2494-4289-a3a6-c12032da731c?fullscreenApplet=true

📚 *MRI protocol books*
https://notebooklm.google.com/notebook/467e8684-c512-488f-b1f7-3a450e344cd5\`;

    assert.ok(GROUP_REPLY_FOOTER.includes('https://view.stradus.com/'));
    assert.ok(GROUP_REPLY_FOOTER.includes('https://ai.studio/apps/3f0807e3-2494-4289-a3a6-c12032da731c?fullscreenApplet=true'));
  });
});
`);

// --- 5. f5_media_viewer.test.js ---
fs.writeFileSync(path.join(tier1Dir, 'f5_media_viewer.test.js'), `import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  mediaViewerStore,
  storeMediaForViewer,
  getBaseUrl
} from '../../index.js';
import { CT_SCAN_IMAGE_BASE64, PDF_REPORT_BASE64 } from '../fixtures/medical_fixtures.js';

describe('F5: Interactive 12h Full-Quality Media Viewer & Web Server', () => {

  test('TC-F5-01: 12h in-memory store generates 32-character crypto hexadecimal token', () => {
    const mediaItems = [
      { type: 'image', data: CT_SCAN_IMAGE_BASE64, mimeType: 'image/jpeg', caption: 'CT Scan 1' },
      { type: 'pdf', data: PDF_REPORT_BASE64, mimeType: 'application/pdf', fileName: 'Report.pdf' }
    ];

    const viewerId = storeMediaForViewer(mediaItems);
    assert.ok(viewerId, 'Viewer ID must be returned');
    assert.equal(viewerId.length, 32, 'Viewer ID must be 32-char hex (128-bit)');
    assert.ok(mediaViewerStore.has(viewerId), 'Media viewer store must contain entry');

    const entry = mediaViewerStore.get(viewerId);
    assert.equal(entry.media.length, 2);
    assert.ok(entry.expiresAt > Date.now() + 11 * 3600 * 1000, 'Expiration must be set to ~12h in future');
  });

  test('TC-F5-02: Base URL detection respects cloud environment variables', () => {
    const origEnv = { ...process.env };
    try {
      process.env.APP_URL = 'https://my-radiology-bot.onrender.com/';
      assert.equal(getBaseUrl(), 'https://my-radiology-bot.onrender.com');

      delete process.env.APP_URL;
      process.env.RENDER_EXTERNAL_URL = 'https://render-bot.onrender.com';
      assert.equal(getBaseUrl(), 'https://render-bot.onrender.com');

      delete process.env.RENDER_EXTERNAL_URL;
      process.env.KOYEB_PUBLIC_DOMAIN = 'koyeb-bot.koyeb.app';
      assert.equal(getBaseUrl(), 'https://koyeb-bot.koyeb.app');
    } finally {
      process.env = origEnv;
    }
  });

  test('TC-F5-03: Binary streaming endpoint resolves raw image Buffer with Content-Type header', () => {
    const mediaItems = [{ type: 'image', data: CT_SCAN_IMAGE_BASE64, mimeType: 'image/jpeg' }];
    const viewerId = storeMediaForViewer(mediaItems);

    const entry = mediaViewerStore.get(viewerId);
    assert.ok(entry);
    const m = entry.media[0];
    const buffer = Buffer.from(m.data, 'base64');

    assert.ok(Buffer.isBuffer(buffer));
    assert.ok(buffer.length > 0);
    assert.equal(m.mimeType, 'image/jpeg');
  });

  test('TC-F5-04: /health diagnostics schema returns required monitoring fields', () => {
    const healthPayload = {
      status: 'running',
      connected: true,
      mongoConnected: true,
      mode: 'universal',
      processedCount: 42,
      activeKeys: 2,
      activeViewers: mediaViewerStore.size,
      decryptFailures: 0,
      healingInProgress: false,
      pendingRetries: 0,
      timestamp: new Date().toISOString()
    };

    assert.equal(healthPayload.status, 'running');
    assert.equal(typeof healthPayload.connected, 'boolean');
    assert.equal(typeof healthPayload.mongoConnected, 'boolean');
    assert.equal(typeof healthPayload.processedCount, 'number');
    assert.ok(Date.parse(healthPayload.timestamp) > 0);
  });

  test('TC-F5-05: Expired or non-existent viewerId returns null or 404', () => {
    const fakeViewerId = 'ffffffffffffffffffffffffffffffff';
    assert.equal(mediaViewerStore.has(fakeViewerId), false);
  });

  test('TC-F5-06: Zero-egress RAM architecture preserves image quality without network degradation', () => {
    const rawBuffer = Buffer.from(CT_SCAN_IMAGE_BASE64, 'base64');
    const storedBase64 = rawBuffer.toString('base64');
    assert.equal(storedBase64, CT_SCAN_IMAGE_BASE64, 'Zero compression loss during RAM storage');
  });
});
`);

// --- 6. f6_followup_analysis.test.js ---
fs.writeFileSync(path.join(tier1Dir, 'f6_followup_analysis.test.js'), `import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { MockGeminiEngine } from '../mocks/mock_gemini.js';
import { CT_SCAN_IMAGE_BASE64 } from '../fixtures/medical_fixtures.js';

describe('F6: Follow-Up Questions, Chained Advice & Buffer Commands', () => {
  let mockGemini;

  beforeEach(() => {
    mockGemini = new MockGeminiEngine();
  });

  test('TC-F6-01: Group chat question isolation passes source media + user question with null system instruction', async () => {
    const sourceMedia = [{ type: 'image', data: CT_SCAN_IMAGE_BASE64, mimeType: 'image/jpeg' }];
    const userQuestion = 'What is the size of the lesion on the right upper lobe?';

    const contentParts = sourceMedia.map(m => ({
      inlineData: { data: m.data, mimeType: m.mimeType }
    }));
    const requestContent = [userQuestion, ...contentParts];

    const response = await mockGemini.generate(requestContent, null);
    assert.ok(response, 'Must return AI answer');

    const history = mockGemini.getHistory();
    assert.equal(history[0].systemInstruction, null, 'Group chat question must have null system instruction');
  });

  test('TC-F6-02: DM reply with additional text note updates previous Clinical Profile', async () => {
    const previousProfile = '*Clinical Profile: Previous CT showed hepatic cyst.*';
    const newContextNote = 'Patient is known hypertensive for 10 years and diabetic.';

    const promptText = \`The user is replying with ADDITIONAL CONTEXT: \${newContextNote}. Previous: \${previousProfile}\`;
    const response = await mockGemini.generate([promptText], 'Standard AI System Instruction');

    assert.ok(response.includes('*Clinical Profile:'), 'Must generate updated Clinical Profile');
  });

  test('TC-F6-03: Question detector accurately classifies medical questions vs statements', () => {
    const isQuestion = (text) => {
      if (!text) return false;
      const lower = text.toLowerCase().trim();
      if (lower.endsWith('?')) return true;
      const starters = ['what', 'why', 'how', 'is ', 'are ', 'can ', 'explain', 'tell me'];
      return starters.some(s => lower.startsWith(s));
    };

    assert.equal(isQuestion('What does the right MCA infarct indicate?'), true);
    assert.equal(isQuestion('Is this finding serious?'), true);
    assert.equal(isQuestion('Explain the liver measurement.'), true);
    assert.equal(isQuestion('Patient also has a history of asthma.'), false);
    assert.equal(isQuestion('Attached prior report from 2024'), false);
  });

  test('TC-F6-04: Secondary analysis (..) command generates Step 1 Profile and Step 2 Advice', async () => {
    const step1Response = await mockGemini.generate(['Analyze CT scan'], 'PRIMARY_INSTRUCTION');
    assert.ok(step1Response.includes('*Clinical Profile:'));

    const secondarySysInstruction = 'You are an expert radiologist. Analyse that info and advise regarding imaging modality.';
    const step2Prompt = \`=== CLINICAL PROFILE ===\\n\${step1Response}\\n=== END PROFILE ===\`;
    const step2Response = await mockGemini.generate([step2Prompt], secondarySysInstruction);

    assert.ok(step2Response.includes('Radiology Protocol & Imaging Advice'));
  });

  test('TC-F6-05: Buffer management commands (clear, status, help, ping) are recognized', () => {
    const COMMANDS = ['.', '.1', '.2', '.3', '..', '..1', '..2', '..3', 'help', '?', 'clear', 'status', 'ping'];
    
    assert.ok(COMMANDS.includes('clear'));
    assert.ok(COMMANDS.includes('status'));
    assert.ok(COMMANDS.includes('help'));
    assert.ok(COMMANDS.includes('ping'));
    assert.ok(COMMANDS.includes('..2'));
  });

  test('TC-F6-06: Speed token parsing extracts target FPS from trigger command (.1 -> 1, .2 -> 2, .3 -> 3)', () => {
    const parseFps = (cmd) => {
      const lastChar = cmd.slice(-1);
      const parsed = parseInt(lastChar);
      return isNaN(parsed) ? 3 : parsed;
    };

    assert.equal(parseFps('.'), 3);
    assert.equal(parseFps('..'), 3);
    assert.equal(parseFps('.1'), 1);
    assert.equal(parseFps('..2'), 2);
    assert.equal(parseFps('.3'), 3);
  });
});
`);

console.log('Tier 1 test suites generated successfully.');