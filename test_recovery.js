// Set environment to test mode before importing index.js
process.env.NODE_ENV = 'test';

import assert from 'assert';
import os from 'os';
import fs from 'fs';

// Import functions and objects from index.js
import {
  CONFIG,
  addToUserBuffer,
  clearUserBuffer,
  getUserBufferCount,
  markUserBufferCompleted,
  markUserBufferFailed,
  runStartupRecovery,
  downloadMediaWithRetry,
  extractFramesFromVideo,
  processMedia,
  activeProcessingUsers,
  setMockSock,
  setDownloadMediaMessageMock,
  setGenerateGeminiContentMock
} from './index.js';

// ======================================================================
// 🛠️ IN-MEMORY MONGOOSE MODEL MOCK FOR SAFE TEST EXECUTION
// ======================================================================
class MockPendingMediaModel {
  constructor() {
    this.docs = [];
  }

  async create(docData) {
    const doc = {
      _id: 'doc_' + Math.random().toString(36).substring(7),
      chatId: docData.chatId,
      senderId: docData.senderId,
      senderName: docData.senderName || '',
      messageId: docData.messageId || 'msg_' + Math.random().toString(36).substring(7),
      type: docData.type,
      data: docData.data || '',
      content: docData.content || '',
      mimeType: docData.mimeType || '',
      caption: docData.caption || '',
      fileName: docData.fileName || '',
      status: docData.status || 'pending',
      processed: docData.processed || false,
      createdAt: docData.createdAt || new Date()
    };
    this.docs.push(doc);
    return doc;
  }

  matchFilter(doc, filter) {
    for (const key of Object.keys(filter)) {
      const val = filter[key];
      if (val && typeof val === 'object' && val.$in) {
        if (!val.$in.includes(doc[key])) return false;
      } else if (val && typeof val === 'object' && val.$ne !== undefined) {
        if (doc[key] === val.$ne) return false;
      } else if (doc[key] !== val) {
        return false;
      }
    }
    return true;
  }

  find(filter = {}) {
    const matches = this.docs.filter(d => this.matchFilter(d, filter));
    return {
      sort: (sortObj) => {
        matches.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
        return Promise.resolve(matches);
      },
      then: (resolve) => resolve(matches)
    };
  }

  async updateMany(filter = {}, update = {}) {
    const matches = this.docs.filter(d => this.matchFilter(d, filter));
    let modifiedCount = 0;
    for (const doc of matches) {
      if (update.$set) {
        Object.assign(doc, update.$set);
      } else {
        Object.assign(doc, update);
      }
      modifiedCount++;
    }
    return { modifiedCount };
  }

  async countDocuments(filter = {}) {
    const matches = this.docs.filter(d => this.matchFilter(d, filter));
    return matches.length;
  }

  async distinct(field, filter = {}) {
    const matches = this.docs.filter(d => this.matchFilter(d, filter));
    const values = new Set();
    for (const doc of matches) {
      if (doc[field] !== undefined) {
        values.add(doc[field]);
      }
    }
    return Array.from(values);
  }
}

import mongoose from 'mongoose';

const mockModel = new MockPendingMediaModel();

// Patch mongoose connection state and model factory
mongoose.connection.readyState = 1;
mongoose.model = (name) => {
  if (name === 'PendingMedia') return mockModel;
  return mockModel;
};

console.log('🧪 Starting Recovery & Retry Mock Test Suite...\n');

let passedTests = 0;
let failedTests = 0;

function logTestPass(name) {
  console.log(`  ✅ PASS: ${name}`);
  passedTests++;
}

function logTestFail(name, error) {
  console.error(`  ❌ FAIL: ${name}`);
  console.error(error);
  failedTests++;
}

// ======================================================================
// TEST 1: SERVER CRASH & STARTUP RECOVERY (TEXT CONTENT PRESERVATION)
// ======================================================================
async function testStartupRecovery() {
  console.log('--------------------------------------------------');
  console.log('Test 1: Server Crash & Startup Recovery Suite');
  console.log('--------------------------------------------------');

  const chatId = 'patient_group_101@g.us';
  const senderId = '919876543210@s.whatsapp.net';
  const senderName = 'Dr. Alice';
  const testTextContent = 'Patient John Doe, 45/M. Presented with severe lower right quadrant abdominal pain, fever, and leukocytosis. Suspected acute appendicitis.';

  // Set up mock AI generator for test
  setGenerateGeminiContentMock(async (requestContent, systemInstruction) => {
    const promptStr = Array.isArray(requestContent) ? requestContent.join('\n') : String(requestContent);
    assert.ok(promptStr.includes(testTextContent), 'AI generator must receive preserved text content');
    return `=== MOCK CLINICAL PROFILE ===\nPatient Summary: John Doe 45M\nImpression: Acute appendicitis suspected.\nSubmitted by: ${senderName}`;
  });

  // Step 1: Simulate abandoned text message saved in MongoDB before crash
  console.log('  [1.1] Simulating abandoned text item in MongoDB buffer...');
  const initialCount = await addToUserBuffer(chatId, senderId, {
    type: 'text',
    content: testTextContent,
    sender: senderName,
    timestamp: Date.now()
  });

  assert.strictEqual(initialCount, 1, 'Buffer count should be 1 after insertion');
  
  const rawDocs = await mockModel.find({ chatId, senderId, processed: false });
  assert.strictEqual(rawDocs.length, 1, 'MongoDB mock should contain 1 unprocessed item');
  assert.strictEqual(rawDocs[0].content, testTextContent, 'Text content must be saved in MongoDB doc');
  assert.strictEqual(rawDocs[0].status, 'pending', 'Initial status should be pending');
  assert.strictEqual(rawDocs[0].processed, false, 'Initial processed flag should be false');
  logTestPass('Abandoned item insertion & schema content verification');

  // Step 2: Set up mock socket and mock AI response
  const sentMessages = [];
  const mockSock = {
    sendMessage: async (jid, content) => {
      sentMessages.push({ jid, content });
      return { key: { id: 'msg_' + Date.now() } };
    },
    sendPresenceUpdate: async () => {}
  };
  setMockSock(mockSock);

  // Step 3: Trigger Startup Recovery Sweep
  console.log('  [1.2] Executing startup recovery sweep...');
  await runStartupRecovery();

  // Step 4: Verify recovery sweep processed items and updated DB state
  const remainingCount = await getUserBufferCount(chatId, senderId);
  assert.strictEqual(remainingCount, 0, 'Pending buffer count should be 0 after recovery sweep');

  const updatedDocs = mockModel.docs.filter(d => d.chatId === chatId && d.senderId === senderId);
  assert.strictEqual(updatedDocs.length, 1, 'Should find 1 document in DB');
  assert.strictEqual(updatedDocs[0].processed, true, 'Recovered document must be marked processed: true');
  assert.strictEqual(updatedDocs[0].status, 'completed', 'Recovered document status must be completed');
  logTestPass('Startup recovery status lifecycle transition (pending -> processing -> completed)');

  // Step 5: Verify text content was preserved in sent message
  assert.strictEqual(sentMessages.length >= 1, true, 'At least 1 response message should be sent to socket');
  const sentText = sentMessages[0].content.text;
  assert.strictEqual(sentText.includes('John Doe 45M') || sentText.includes('MOCK CLINICAL PROFILE'), true, 'Sent message must contain generated profile');
  logTestPass('Text content preserved and processed cleanly without data loss');
}

// ======================================================================
// TEST 2: DROPPED CDN CONNECTION & EXPONENTIAL BACKOFF
// ======================================================================
async function testExponentialBackoff() {
  console.log('\n--------------------------------------------------');
  console.log('Test 2: Dropped CDN Connection & Exponential Backoff');
  console.log('--------------------------------------------------');

  // Case 2a: Transient network drops with successful recovery after 2 retries
  console.log('  [2.1] Testing transient CDN drop with recovery after 2 retries...');
  let attemptsMade = 0;
  const attemptTimestamps = [];

  const mockDownloadSuccessAfterRetries = async (msg, type, options, ctx) => {
    attemptsMade++;
    attemptTimestamps.push(Date.now());
    if (attemptsMade < 3) {
      throw new Error(`CDN socket timeout on attempt ${attemptsMade}`);
    }
    return Buffer.from('mock_media_binary_data');
  };

  setDownloadMediaMessageMock(mockDownloadSuccessAfterRetries);

  const fakeMsg = { key: { remoteJid: '12345@s.whatsapp.net', id: 'msg_media_1' } };
  const resultBuffer = await downloadMediaWithRetry(fakeMsg, 5, 100);

  assert.ok(resultBuffer, 'Downloaded buffer should not be null');
  assert.strictEqual(resultBuffer.toString(), 'mock_media_binary_data', 'Downloaded buffer content match');
  assert.strictEqual(attemptsMade, 3, 'Should take exactly 3 attempts');

  // Verify exponential backoff intervals
  const delay1 = attemptTimestamps[1] - attemptTimestamps[0]; // 100 * 2^1 = 200ms approx
  const delay2 = attemptTimestamps[2] - attemptTimestamps[1]; // 100 * 2^2 = 400ms approx

  assert.ok(delay1 >= 150, `Delay 1 should be ~200ms (got ${delay1}ms)`);
  assert.ok(delay2 >= 300, `Delay 2 should be ~400ms (got ${delay2}ms)`);
  assert.ok(delay2 > delay1, `Exponential backoff verified: delay2 (${delay2}ms) > delay1 (${delay1}ms)`);
  logTestPass('Transient network drops retry succeeded with exponential backoff delays');

  // Case 2b: Permanent CDN failure exceeding max retries
  console.log('  [2.2] Testing permanent CDN failure exceeding max retries...');
  let totalFailAttempts = 0;

  const mockDownloadAlwaysFail = async () => {
    totalFailAttempts++;
    throw new Error('Connection refused by WhatsApp CDN server');
  };

  setDownloadMediaMessageMock(mockDownloadAlwaysFail);

  let caughtError = null;
  try {
    await downloadMediaWithRetry(fakeMsg, 3, 50);
  } catch (err) {
    caughtError = err;
  }

  assert.ok(caughtError, 'Should throw error when max retries exceeded');
  assert.strictEqual(totalFailAttempts, 3, 'Should attempt maxRetries (3) times before failing');
  assert.strictEqual(caughtError.message, 'Connection refused by WhatsApp CDN server', 'Error message preserved');
  logTestPass('Permanent CDN failure handled gracefully after max retries');
}

// ======================================================================
// TEST 3: CONCURRENCY LOCK & TEMP FILE CLEANUP
// ======================================================================
async function testConcurrencyAndCleanup() {
  console.log('\n--------------------------------------------------');
  console.log('Test 3: Concurrency Lock & Resource Cleanup Suite');
  console.log('--------------------------------------------------');

  // Case 3a: Per-user mutex lock prevents duplicate concurrent execution
  console.log('  [3.1] Testing per-user concurrency mutex lock...');
  const chatId = 'lock_test_chat@g.us';
  const senderId = 'user_lock_99@s.whatsapp.net';
  const userLockKey = `${chatId}:${senderId}`;

  activeProcessingUsers.add(userLockKey);

  let secondCallExecuted = false;
  const mockSock = {
    sendMessage: async () => { secondCallExecuted = true; }
  };

  // Attempt concurrent call while lock is active
  await processMedia(mockSock, chatId, [{ type: 'text', content: 'test', sender: 'user' }], false, null, senderId, 'UserLock');

  assert.strictEqual(secondCallExecuted, false, 'Concurrent call must be blocked when lock is held');
  activeProcessingUsers.delete(userLockKey);
  logTestPass('Per-user mutex lock blocks concurrent processMedia calls');

  // Case 3b: Automatic lock release upon processMedia completion
  console.log('  [3.2] Testing automatic mutex lock release on processMedia completion...');
  const autoLockChatId = 'auto_lock_chat@g.us';
  const autoLockSenderId = 'user_auto_lock_100@s.whatsapp.net';
  const autoUserLockKey = `${autoLockChatId}:${autoLockSenderId}`;

  let processExecuted = false;
  let lockHeldDuringProcessing = false;
  const autoMockSock = {
    sendMessage: async () => {
      processExecuted = true;
      lockHeldDuringProcessing = activeProcessingUsers.has(autoUserLockKey);
      return { key: { id: 'msg_auto_1' } };
    },
    sendPresenceUpdate: async () => {}
  };
  setMockSock(autoMockSock);

  setGenerateGeminiContentMock(async () => '=== MOCK CLINICAL PROFILE ===');

  await processMedia(autoMockSock, autoLockChatId, [{ type: 'text', content: 'auto lock test', sender: 'UserAutoLock' }], false, null, autoLockSenderId, 'UserAutoLock');

  assert.strictEqual(processExecuted, true, 'processMedia should execute successfully');
  assert.strictEqual(lockHeldDuringProcessing, true, 'Lock should be held during processMedia execution');
  assert.strictEqual(activeProcessingUsers.has(autoUserLockKey), false, 'Lock must be released automatically after processMedia completes execution');
  logTestPass('Automatic mutex lock release on processMedia completion verified');

  // Case 3c: Automatic lock release on processMedia failure
  console.log('  [3.3] Testing automatic mutex lock release on processMedia failure...');
  const errLockChatId = 'err_lock_chat@g.us';
  const errLockSenderId = 'user_err_lock_101@s.whatsapp.net';
  const errUserLockKey = `${errLockChatId}:${errLockSenderId}`;

  const errMockSock = {
    sendMessage: async () => { return { key: { id: 'msg_err' } }; },
    sendPresenceUpdate: async () => {}
  };
  setMockSock(errMockSock);

  setGenerateGeminiContentMock(async () => {
    assert.strictEqual(activeProcessingUsers.has(errUserLockKey), true, 'Lock should be held when error occurs in generator');
    throw new Error('Simulated Gemini API Failure');
  });

  await processMedia(errMockSock, errLockChatId, [{ type: 'text', content: 'err test', sender: 'UserErrLock' }], false, null, errLockSenderId, 'UserErrLock', null, 3, false, null, 1);

  assert.strictEqual(activeProcessingUsers.has(errUserLockKey), false, 'Lock must be released automatically even when processMedia encounters an error');
  logTestPass('Automatic mutex lock release on processMedia failure verified');

  // Case 3d: Temp file leak prevention check
  console.log('  [3.4] Testing temp file leak prevention in extractFramesFromVideo...');
  const fakeVideoBuffer = Buffer.from('invalid_video_stream');
  let frameError = null;
  try {
    await extractFramesFromVideo(fakeVideoBuffer, 1);
  } catch (err) {
    frameError = err;
  }

  assert.ok(frameError, 'Invalid video buffer should throw ffmpeg error');
  
  // Verify no orphaned input_*.mp4 files remain in os.tmpdir()
  const tmpFiles = fs.readdirSync(os.tmpdir());
  const residualInputFiles = tmpFiles.filter(f => f.startsWith('input_') && f.endsWith('.mp4'));
  assert.strictEqual(residualInputFiles.length, 0, 'No temporary video input files should be leaked in tmpdir');
  logTestPass('Temp video files cleaned up cleanly on error');
}

// ======================================================================
// RUN ALL TESTS
// ======================================================================
(async () => {
  try {
    await testStartupRecovery();
    await testExponentialBackoff();
    await testConcurrencyAndCleanup();

    console.log('\n==================================================');
    console.log(`SUMMARY: ${passedTests} passed, ${failedTests} failed.`);
    console.log('==================================================\n');

    if (failedTests > 0) {
      process.exit(1);
    } else {
      console.log('🎉 ALL RECOVERY & BACKOFF TESTS PASSED SUCCESSFULLY! (exit code 0)\n');
      process.exit(0);
    }
  } catch (err) {
    console.error('\n💥 Unexpected Test Failure:', err);
    process.exit(1);
  }
})();
