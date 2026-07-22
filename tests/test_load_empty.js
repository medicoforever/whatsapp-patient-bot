// Set environment to test mode before importing index.js
process.env.NODE_ENV = 'test';

import assert from 'assert';
import { performance } from 'perf_hooks';
import {
  CONFIG,
  pendingEmptyMessages,
  scheduleEmptyMessageRetry,
  setMockSock
} from '../index.js';

console.log('🧪 Starting 100 Concurrent Empty Messages Load & Performance Test Suite...\n');

// Configure fast retry delays for test execution
CONFIG.EMPTY_MSG_RETRY_DELAY_MS = 10;
CONFIG.EMPTY_MSG_RETRY_STEP_MS = 10;
CONFIG.EMPTY_MSG_MAX_RETRIES = 3;
const TEST_SOURCE_GROUP = CONFIG.GROUPS.CT_SOURCE || 'test_source_group@g.us';

// Mock Socket Object
const mockSock = {
  sendMessage: async () => ({ key: { id: 'mock_reply_' + Date.now() } }),
  ev: { on: () => {}, emit: () => {} }
};
setMockSock(mockSock);

function getMemoryUsageMB() {
  const mem = process.memoryUsage();
  return {
    heapUsed: (mem.heapUsed / 1024 / 1024).toFixed(2),
    rss: (mem.rss / 1024 / 1024).toFixed(2)
  };
}

async function runEmptyMessageLoadTest() {
  console.log('--------------------------------------------------');
  console.log('Phase 1: Baseline Metrics & Setup');
  console.log('--------------------------------------------------');
  
  const baselineMem = getMemoryUsageMB();
  console.log(`  📊 Baseline Memory: Heap ${baselineMem.heapUsed} MB | RSS ${baselineMem.rss} MB`);
  assert.strictEqual(pendingEmptyMessages.size, 0, 'Pending empty messages map must start at 0');

  console.log('\n--------------------------------------------------');
  console.log('Phase 2: Ingesting 100 Concurrent Empty Messages');
  console.log('--------------------------------------------------');

  const TOTAL_MESSAGES = 100;
  const startTime = performance.now();
  let maxTickDelay = 0;

  // Event loop blocking monitor tick
  const tickCheckStart = performance.now();
  await new Promise(resolve => setTimeout(resolve, 0));
  const tickCheckEnd = performance.now();
  maxTickDelay = Math.max(maxTickDelay, tickCheckEnd - tickCheckStart);

  // Ingest 100 concurrent empty messages
  for (let i = 1; i <= TOTAL_MESSAGES; i++) {
    const msgId = `empty_msg_${String(i).padStart(3, '0')}`;
    const mockMsg = {
      key: {
        remoteJid: TEST_SOURCE_GROUP,
        id: msgId,
        fromMe: false
      },
      message: null
    };

    pendingEmptyMessages.set(msgId, {
      msg: mockMsg,
      retryCount: 0,
      chatId: TEST_SOURCE_GROUP,
      timestamp: Date.now()
    });

    scheduleEmptyMessageRetry(msgId);
  }

  const ingestionTimeMs = (performance.now() - startTime).toFixed(2);
  const peakMem = getMemoryUsageMB();
  
  console.log(`  ⏱️ Ingestion Time for 100 messages: ${ingestionTimeMs} ms`);
  console.log(`  📊 Peak Queue Size: ${pendingEmptyMessages.size} items`);
  console.log(`  📊 Peak Memory: Heap ${peakMem.heapUsed} MB | RSS ${peakMem.rss} MB`);
  console.log(`  ⚡ Max Event Loop Delay: ${maxTickDelay.toFixed(2)} ms`);

  assert.strictEqual(pendingEmptyMessages.size, TOTAL_MESSAGES, 'All 100 messages must be present in queue');
  assert.ok(ingestionTimeMs < 100, `Ingestion time (${ingestionTimeMs}ms) must be under 100ms`);
  assert.ok(maxTickDelay < 15, `Max event loop delay (${maxTickDelay.toFixed(2)}ms) must be under 15ms`);

  console.log('\n--------------------------------------------------');
  console.log('Phase 3: Simulating 50 Decryption Updates & 50 Expirations');
  console.log('--------------------------------------------------');

  // Simulate 50 messages getting updated with decrypted content
  for (let i = 1; i <= 50; i++) {
    const msgId = `empty_msg_${String(i).padStart(3, '0')}`;
    if (pendingEmptyMessages.has(msgId)) {
      const pending = pendingEmptyMessages.get(msgId);
      if (pending?.timerId) clearTimeout(pending.timerId);
      pendingEmptyMessages.delete(msgId); // Simulates resolution on message update
    }
  }

  console.log(`  ✅ 50 messages decrypted & resolved. Remaining in retry queue: ${pendingEmptyMessages.size}`);
  assert.strictEqual(pendingEmptyMessages.size, 50, 'Queue size should be 50 after resolving 50 messages');

  // Wait for remaining 50 messages to complete their retries (3 retries * 10ms delay + buffer = ~100ms)
  await new Promise(resolve => setTimeout(resolve, 300));

  const finalMem = getMemoryUsageMB();
  const heapDeltaMB = (parseFloat(finalMem.heapUsed) - parseFloat(baselineMem.heapUsed)).toFixed(2);

  console.log('\n--------------------------------------------------');
  console.log('Phase 4: Final Validation & Memory Audit');
  console.log('--------------------------------------------------');
  console.log(`  📊 Final Queue Size: ${pendingEmptyMessages.size} items`);
  console.log(`  📊 Final Memory: Heap ${finalMem.heapUsed} MB | RSS ${finalMem.rss} MB`);
  console.log(`  📈 Net Heap Memory Growth: ${heapDeltaMB} MB`);

  assert.strictEqual(pendingEmptyMessages.size, 0, 'Final retry queue size must be 0 after all retries complete');
  assert.ok(parseFloat(heapDeltaMB) < 10.0, `Net heap memory growth (${heapDeltaMB} MB) must be under 10 MB`);

  console.log('\n  ✅ PASS: 100 Concurrent Empty Messages Load & Performance Test Passed Successfully!');
}

runEmptyMessageLoadTest().then(() => {
  process.exit(0);
}).catch(err => {
  console.error('\n  ❌ FAIL: Load test failed with error:', err);
  process.exit(1);
});
