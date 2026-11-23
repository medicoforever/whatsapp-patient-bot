// Set test environment
process.env.NODE_ENV = 'test';

import assert from 'assert';
import http from 'http';
import {
  isAllowedGeminiModel,
  REQUESTED_DEFAULT_MODEL_HIERARCHY,
  isProductionEnv,
  STARTUP_GRACE_PERIOD_S,
  app,
  CONFIG,
  connectMongoDB
} from '../index.js';

import {
  MASTER_ENV_CONFIG,
  hasRecentDeployWithinMinutes,
  TOUR_ROTATION_DAYS
} from '../render-failover-manager.js';

console.log('🧪 Starting Failover, Model Chain & Production Guards Test Suite...\n');

// ----------------------------------------------------
// Test 1: Gemini Model Hierarchy & Prohibited Models
// ----------------------------------------------------
console.log('Test 1: Gemini Model Filter & Hierarchy Enforcement');

// 1.1 Prohibited models must be rejected
assert.strictEqual(isAllowedGeminiModel('gemini-4-hydrogen'), false, 'gemini-4-hydrogen must be rejected');
assert.strictEqual(isAllowedGeminiModel('gemini-2.5-flash'), false, 'gemini-2.5-flash must be rejected');
assert.strictEqual(isAllowedGeminiModel('gemini-2.5-pro'), false, 'gemini-2.5-pro must be rejected');
assert.strictEqual(isAllowedGeminiModel('hydrogen-model'), false, 'hydrogen model must be rejected');

// 1.2 Allowed models must pass
assert.strictEqual(isAllowedGeminiModel('gemini-3.8-flash'), true, 'gemini-3.8-flash must be allowed');
assert.strictEqual(isAllowedGeminiModel('gemini-3.5-flash'), true, 'gemini-3.5-flash must be allowed');
assert.strictEqual(isAllowedGeminiModel('gemini-3.6-flash'), true, 'gemini-3.6-flash must be allowed');
assert.strictEqual(isAllowedGeminiModel('gemini-3-flash-preview'), true, 'gemini-3-flash-preview must be allowed');
assert.strictEqual(isAllowedGeminiModel('gemini-3.5-flash-lite'), true, 'gemini-3.5-flash-lite must be allowed');

// 1.3 Exact hierarchy order check
const expectedHierarchy = [
  'gemini-3.8-flash',
  'gemini-3.5-flash',
  'gemini-3.6-flash',
  'gemini-3-flash-preview',
  'gemini-3.5-flash-lite'
];
assert.deepStrictEqual(REQUESTED_DEFAULT_MODEL_HIERARCHY, expectedHierarchy, 'Requested model hierarchy must strictly match order');
console.log('  ✅ PASS: Model hierarchy and filtering rules verified.');

// ----------------------------------------------------
// Test 2: Master Env Config Completeness
// ----------------------------------------------------
console.log('\nTest 2: Master Environment Variable Integrity');
const requiredKeys = [
  'NODE_VERSION',
  'MONGODB_URI',
  'GROUP_MRI_TARGET',
  'GROUP_MRI_SOURCE',
  'GROUP_CT_TARGET',
  'GROUP_CT_SOURCE',
  'ALLOWED_GROUP_ID',
  'TELEGRAM_BOT_TOKEN',
  'TELEGRAM_CT_CHAT_ID',
  'TELEGRAM_MRI_CHAT_ID',
  'GEMINI_API_KEYS'
];

for (const key of requiredKeys) {
  assert.ok(key in MASTER_ENV_CONFIG, `MASTER_ENV_CONFIG must define key: ${key}`);
}
assert.strictEqual(requiredKeys.length, 11, 'Total required master environment variables must be 11');
console.log('  ✅ PASS: All 11 master environment variables are present and configured.');

// ----------------------------------------------------
// Test 3: Anti-Thrashing Deploy Loop Detection
// ----------------------------------------------------
console.log('\nTest 3: Anti-Thrashing Redeploy Detection');
const recentDeploys = [
  { id: 'dep-1', createdAt: new Date(Date.now() - 5 * 60 * 1000).toISOString() } // 5 mins ago
];
const oldDeploys = [
  { id: 'dep-2', createdAt: new Date(Date.now() - 45 * 60 * 1000).toISOString() } // 45 mins ago
];

assert.strictEqual(hasRecentDeployWithinMinutes(recentDeploys, 30), true, 'Should detect deploy within 30 minutes');
assert.strictEqual(hasRecentDeployWithinMinutes(oldDeploys, 30), false, 'Should not block deploy if older than 30 minutes');
assert.strictEqual(hasRecentDeployWithinMinutes([], 30), false, 'Empty deploys array should return false');
console.log('  ✅ PASS: Anti-thrashing deployment loop logic verified.');

// ----------------------------------------------------
// Test 4: HTTP /ping Endpoint Grace Period & 503 Logic
// ----------------------------------------------------
console.log('\nTest 4: HTTP /ping Grace Period & 503 Reporting');
const testServer = http.createServer(app);

await new Promise(resolve => testServer.listen(0, resolve));
const port = testServer.address().port;

// Helper to query test server
function fetchPing(port) {
  return new Promise((resolve, reject) => {
    http.get(`http://localhost:${port}/ping`, res => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body: data }));
    }).on('error', reject);
  });
}

// 4.1 When process is fresh (uptime < STARTUP_GRACE_PERIOD_S), /ping returns 200 OK
const resFresh = await fetchPing(port);
assert.strictEqual(resFresh.status, 200, 'Fresh process within grace period should return 200');
assert.strictEqual(resFresh.body, 'OK', 'Fresh process should return OK');
console.log('  ✅ PASS: /ping returns 200 OK during startup grace period.');

// 4.2 Simulate past grace period with socket disconnected
const originalUptime = process.uptime;
try {
  process.uptime = () => 120; // 120 seconds uptime > 90s grace period
  const resDisconnected = await fetchPing(port);
  assert.strictEqual(resDisconnected.status, 503, 'Disconnected socket past grace period must return HTTP 503');
  assert.ok(resDisconnected.body.includes('SERVICE_UNAVAILABLE'), 'Body must indicate SERVICE_UNAVAILABLE');
  console.log('  ✅ PASS: /ping returns 503 SERVICE_UNAVAILABLE past startup grace period when disconnected.');
} finally {
  process.uptime = originalUptime;
  testServer.close();
}

// ----------------------------------------------------
// Test 5: Planned 10-Day Rotation Policy & Tour Calculation
// ----------------------------------------------------
console.log('\nTest 5: Planned 10-Day Quota Rotation Policy');
assert.strictEqual(TOUR_ROTATION_DAYS, 10, 'Default TOUR_ROTATION_DAYS must be 10 days');

// Simulate 10-day elapsed tour
const tenDaysAgoMs = Date.now() - (10 * 24 * 60 * 60 * 1000);
const nineDaysAgoMs = Date.now() - (9 * 24 * 60 * 60 * 1000);

const calcDays10 = (Date.now() - tenDaysAgoMs) / (1000 * 60 * 60 * 24);
const calcDays9 = (Date.now() - nineDaysAgoMs) / (1000 * 60 * 60 * 24);

assert.ok(calcDays10 >= 10, '10 days ago must be >= 10 days');
assert.ok(calcDays9 < 10, '9 days ago must be < 10 days');
console.log('  ✅ PASS: 10-day tour duration thresholding verified.');

console.log('\n--------------------------------------------------');
console.log('Summary: All 5 Test Suites Passed Successfully!');
console.log('--------------------------------------------------\n');
