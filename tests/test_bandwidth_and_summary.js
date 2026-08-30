// Set environment to test mode before importing index.js
process.env.NODE_ENV = 'test';

import assert from 'assert';
import { formatInputMediaSummary, optimizeImageForAi, processMedia, setMockSock, setGenerateGeminiContentMock } from '../index.js';

console.log('🧪 Starting Bandwidth Optimization & Media Summary Test Suite...\n');

let passed = 0;
let failed = 0;

function runTest(description, fn) {
  try {
    fn();
    console.log(`  ✅ PASS: ${description}`);
    passed++;
  } catch (err) {
    console.error(`  ❌ FAIL: ${description}`);
    console.error(`     Error: ${err.message}`);
    failed++;
  }
}

async function runAsyncTest(description, fn) {
  try {
    await fn();
    console.log(`  ✅ PASS: ${description}`);
    passed++;
  } catch (err) {
    console.error(`  ❌ FAIL: ${description}`);
    console.error(`     Error: ${err.message}`);
    failed++;
  }
}

// ----------------------------------------------------------------------
// 1. Deterministic Media Summary Formatting Tests
// ----------------------------------------------------------------------

runTest('Single Image formatting', () => {
  const result = formatInputMediaSummary({ images: 1, pdfs: 0, audio: 0, video: 0, texts: 0 });
  assert.strictEqual(result, '📥 *Input Media:* 1 Image');
});

runTest('Multiple Images formatting', () => {
  const result = formatInputMediaSummary({ images: 5, pdfs: 0, audio: 0, video: 0, texts: 0 });
  assert.strictEqual(result, '📥 *Input Media:* 5 Images');
});

runTest('Mixed Media (3 Images, 1 PDF, 2 Audio)', () => {
  const result = formatInputMediaSummary({ images: 3, pdfs: 1, audio: 2, video: 0, texts: 0 });
  assert.strictEqual(result, '📥 *Input Media:* 3 Images, 1 PDF, 2 Audio Notes');
});

runTest('All Media Types (2 Images, 2 PDFs, 1 Audio, 1 Video, 1 Text)', () => {
  const result = formatInputMediaSummary({ images: 2, pdfs: 2, audio: 1, video: 1, texts: 1 });
  assert.strictEqual(result, '📥 *Input Media:* 2 Images, 2 PDFs, 1 Audio Note, 1 Video, 1 Text Note');
});

runTest('Empty Media formatting', () => {
  const result = formatInputMediaSummary({ images: 0, pdfs: 0, audio: 0, video: 0, texts: 0 });
  assert.strictEqual(result, '📥 *Input Media:* None');
});

// ----------------------------------------------------------------------
// 2. Image Compression Functionality Tests
// ----------------------------------------------------------------------

await runAsyncTest('optimizeImageForAi skips small images (<250KB)', async () => {
  const smallBase64 = Buffer.from('small dummy data').toString('base64');
  const result = await optimizeImageForAi(smallBase64, 'image/jpeg');
  assert.strictEqual(result.data, smallBase64);
  assert.strictEqual(result.mimeType, 'image/jpeg');
});

await runAsyncTest('optimizeImageForAi skips non-image mime types', async () => {
  const dummyBase64 = Buffer.alloc(500000, 1).toString('base64');
  const result = await optimizeImageForAi(dummyBase64, 'application/pdf');
  assert.strictEqual(result.data, dummyBase64);
  assert.strictEqual(result.mimeType, 'application/pdf');
});

// ----------------------------------------------------------------------
// 3. End-to-End processMedia output format test
// ----------------------------------------------------------------------

await runAsyncTest('processMedia includes deterministic media summary and source viewer link in group chats', async () => {
  const sentMessages = [];
  const mockSock = {
    sendMessage: async (jid, content) => {
      sentMessages.push({ jid, content });
      return { key: { id: 'test_msg_123' } };
    }
  };
  setMockSock(mockSock);

  setGenerateGeminiContentMock(async () => {
    return '*Clinical Profile: Test clinical profile summary.*';
  });

  const testMedia = [
    { type: 'image', data: Buffer.from('img1').toString('base64'), mimeType: 'image/jpeg', caption: '' },
    { type: 'image', data: Buffer.from('img2').toString('base64'), mimeType: 'image/jpeg', caption: '' },
    { type: 'pdf', data: Buffer.from('pdf1').toString('base64'), mimeType: 'application/pdf', caption: '' }
  ];

  await processMedia(mockSock, '120363405051771143@g.us', testMedia, false, null, '919876543210@s.whatsapp.net', 'Test Doctor');

  assert.strictEqual(sentMessages.length, 1, 'Should have sent 1 message');
  const text = sentMessages[0].content.text;

  // Verify Media Summary is present
  assert.ok(text.includes('📥 *Input Media:* 2 Images, 1 PDF'), 'Message must contain deterministic media summary');

  // Verify Source Media viewer link exists in group
  assert.ok(text.includes('/view/'), 'Message must contain /view/ link in group chat');
  assert.ok(text.includes('Source Media'), 'Message must contain Source Media link in group chat');
});

console.log('\n--------------------------------------------------');
console.log(`Summary: ${passed} Passed, ${failed} Failed`);
console.log('--------------------------------------------------');

process.exit(failed > 0 ? 1 : 0);
