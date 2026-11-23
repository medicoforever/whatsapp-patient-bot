/**
 * 24/7 Render Failover & Uptime Manager
 * 
 * Logic:
 * 1. Checks all 5 Render accounts sequentially with fallback defaults.
 * 2. Dynamically fetches/verifies service IDs and live URLs from Render API.
 * 3. Classifies accounts: ACTIVE, STANDBY, BILLING_EXHAUSTED (hours limit).
 * 4. Checks active service HTTP /health, socket connection, and graceful in-memory recovery.
 * 5. If active service is unhealthy or suspended, seamlessly fails over to the best healthy standby account.
 * 6. Puts standby accounts to sleep (suspend) to preserve their 750 free instance hours.
 * 7. Updates cron-job.org keep-alive ping and watchdog dispatchers.
 */

import { fileURLToPath } from 'url';

const CRON_JOB_API_KEY = (process.env.CRON_JOB_API_KEY || '').trim() || '3GKdFNCZXErgSKSCeHMXG2SGrBYzGN6pldcaHqBHdb8=';
const CRON_JOB_ID = (process.env.CRON_JOB_ID || '').trim() || '7156467';
const CRON_JOB_WATCHDOG_ID = (process.env.CRON_JOB_WATCHDOG_ID || '').trim() || '8538883';
const GITHUB_TOKEN = (process.env.GITHUB_TOKEN || process.env.GH_TOKEN || '').trim();
const TOUR_ROTATION_DAYS = parseFloat(process.env.TOUR_ROTATION_DAYS || '10');

const RENDER_ACCOUNTS = [
  {
    name: 'medicoforever003',
    apiKey: (process.env.RENDER_API_KEY_003 || process.env.RENDER_API_KEY_4 || '').trim() || 'rnd_k7763fBzTQHt7R0crVsSQrWYoKFB',
    serviceId: (process.env.RENDER_SERVICE_ID_003 || process.env.RENDER_SERVICE_ID_4 || '').trim() || 'srv-da46e0fqj5pc73bdboqg',
    url: 'https://whatsapp-patient-bot-zcmf.onrender.com'
  },
  {
    name: 'medicoforever008',
    apiKey: (process.env.RENDER_API_KEY_008 || process.env.RENDER_API_KEY_1 || '').trim() || 'rnd_NNtHMoAwbdGttv0F4WDvFqnbg8bR',
    serviceId: (process.env.RENDER_SERVICE_ID_008 || process.env.RENDER_SERVICE_ID_1 || '').trim() || 'srv-daa4ugpf2nfc739834v0',
    url: 'https://whatsapp-patient-bot-f9lc.onrender.com'
  },
  {
    name: 'medicoforever002',
    apiKey: (process.env.RENDER_API_KEY_002 || process.env.RENDER_API_KEY_2 || '').trim() || 'rnd_jYutYuSK6dtAZwYKFbrjI1ekhffB',
    serviceId: (process.env.RENDER_SERVICE_ID_002 || process.env.RENDER_SERVICE_ID_2 || '').trim() || 'srv-d5jatbq4d50c73fpbgcg',
    url: 'https://whatsapp-patient-bot.onrender.com'
  },
  {
    name: 'raddoc1996',
    apiKey: (process.env.RENDER_API_KEY_RADDOC || process.env.RENDER_API_KEY_3 || '').trim() || 'rnd_ATJs45AaaYcnkL3SETD3vBdkWVmf',
    serviceId: (process.env.RENDER_SERVICE_ID_RADDOC || process.env.RENDER_SERVICE_ID_3 || '').trim() || 'srv-d9uvkuvavr4c73bljb10',
    url: 'https://whatsapp-patient-bot-b4tl.onrender.com'
  },
  {
    name: 'medicoforever006',
    apiKey: (process.env.RENDER_API_KEY_006 || process.env.RENDER_API_KEY_5 || '').trim() || 'rnd_AFsAqE0AQo1AqPFhY56LVYCbvlE9',
    serviceId: (process.env.RENDER_SERVICE_ID_006 || process.env.RENDER_SERVICE_ID_5 || '').trim() || 'srv-dbafg0e1egvs73agvsk0',
    url: 'https://whatsapp-patient-bot-pmfg.onrender.com'
  }
];

const MASTER_ENV_CONFIG = {
  NODE_VERSION: '20.10.0',
  MONGODB_URI: (process.env.MONGODB_URI || '').trim() || 'mongodb+srv://raddoc1996_db_user:lNryCi4OV8MApRoC@cluster0.yxtnjug.mongodb.net/?appName=Cluster0',
  GROUP_MRI_TARGET: (process.env.GROUP_MRI_TARGET || '').trim() || '120363424585068867@g.us',
  GROUP_MRI_SOURCE: (process.env.GROUP_MRI_SOURCE || '').trim() || '120363047547742844@g.us',
  GROUP_CT_TARGET: (process.env.GROUP_CT_TARGET || '').trim() || '120363425099016220@g.us',
  GROUP_CT_SOURCE: (process.env.GROUP_CT_SOURCE || '').trim() || '919442508568-1594612808@g.us',
  ALLOWED_GROUP_ID: (process.env.ALLOWED_GROUP_ID || '').trim() || '120363405051771143@g.us',
  TELEGRAM_BOT_TOKEN: (process.env.TELEGRAM_BOT_TOKEN || '').trim() || '8858327878:AAFuXpgwGN-vbDf-i5dlzwX46-D75BMMKEo',
  TELEGRAM_CT_CHAT_ID: (process.env.TELEGRAM_CT_CHAT_ID || '').trim() || '-1004456617705',
  TELEGRAM_MRI_CHAT_ID: (process.env.TELEGRAM_MRI_CHAT_ID || '').trim() || '-1004380868476',
  GEMINI_API_KEYS: (process.env.GEMINI_API_KEYS || '').trim()
};

async function getAccountEnvVars(account) {
  if (!account.apiKey || !account.serviceId) return null;
  try {
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}/env-vars`, {
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json'
      }
    });
    if (res.ok) {
      const data = await res.json();
      const map = new Map();
      if (Array.isArray(data)) {
        for (const item of data) {
          const k = item.envVar?.key || item.key;
          const v = item.envVar?.value ?? item.value ?? '';
          if (k) map.set(k, v);
        }
      }
      return map;
    }
  } catch (err) {
    console.warn(`[Failover] Failed fetching env vars for ${account.name}: ${err.message}`);
  }
  return null;
}

async function setAccountTourStart(account, isoTimestamp) {
  if (!account.apiKey || !account.serviceId) return false;
  try {
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}/env-vars`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify([
        { key: 'ACTIVE_TOUR_START', value: isoTimestamp }
      ])
    });
    return res.ok;
  } catch (err) {
    console.warn(`[Failover] Failed setting ACTIVE_TOUR_START on ${account.name}: ${err.message}`);
    return false;
  }
}

async function getActiveTourDays(account, service, uptimeSeconds = 0) {
  try {
    const envMap = await getAccountEnvVars(account);
    const tourStartStr = envMap?.get('ACTIVE_TOUR_START');
    if (tourStartStr) {
      const tourStartMs = Date.parse(tourStartStr);
      if (!isNaN(tourStartMs) && tourStartMs > 0 && tourStartMs <= Date.now()) {
        const days = (Date.now() - tourStartMs) / (1000 * 60 * 60 * 24);
        return days;
      }
    }

    // If ACTIVE_TOUR_START is not set yet, initialize from service.updatedAt or current uptime
    let initialDate = new Date();
    if (uptimeSeconds && uptimeSeconds > 0) {
      initialDate = new Date(Date.now() - (uptimeSeconds * 1000));
    } else if (service?.updatedAt) {
      const parsedUpdated = Date.parse(service.updatedAt);
      if (!isNaN(parsedUpdated) && parsedUpdated > 0) {
        initialDate = new Date(parsedUpdated);
      }
    }

    console.log(`[Failover] 📌 Initializing ACTIVE_TOUR_START for ${account.name} as ${initialDate.toISOString()}`);
    await setAccountTourStart(account, initialDate.toISOString());

    return (Date.now() - initialDate.getTime()) / (1000 * 60 * 60 * 24);
  } catch (err) {
    console.warn(`[Failover] Warning: could not parse tour start for ${account.name}: ${err.message}`);
    return (uptimeSeconds || 0) / (60 * 60 * 24);
  }
}

async function resolveMasterEnvConfig() {
  const master = { ...MASTER_ENV_CONFIG };

  // Allow process.env overrides
  for (const key of Object.keys(master)) {
    if (process.env[key] && process.env[key].trim()) {
      master[key] = process.env[key].trim();
    }
  }

  // Scan reachable Render accounts to pull configured variables (e.g. GEMINI_API_KEYS, MONGODB_URI)
  for (const acc of RENDER_ACCOUNTS) {
    if (!acc.apiKey || !acc.serviceId) continue;
    try {
      const map = await getAccountEnvVars(acc);
      if (map && map.size >= 1) {
        for (const [k, v] of map.entries()) {
          if (v && (!master[k] || master[k].length === 0)) {
            master[k] = v;
          }
        }
      }
    } catch (_) {}
  }

  return master;
}

async function syncAccountEnvVars(account, masterConfig) {
  if (!account.apiKey || !account.serviceId) {
    return { synced: false, reason: 'missing_credentials' };
  }

  const existingMap = await getAccountEnvVars(account);
  if (!existingMap) {
    return { synced: false, reason: 'failed_to_fetch_existing_vars' };
  }

  const requiredKeys = Object.keys(masterConfig);
  const missingOrEmpty = [];

  for (const key of requiredKeys) {
    const existingVal = existingMap.get(key);
    if (existingVal === undefined || existingVal === null || existingVal === '') {
      missingOrEmpty.push(key);
    }
  }

  if (missingOrEmpty.length === 0) {
    console.log(`[Failover] 🛡️ Configuration on ${account.name} is complete (${existingMap.size} variables verified).`);
    return { synced: true, updated: false, missingCount: 0 };
  }

  console.warn(`[Failover] ⚠️ Self-Healing Triggered: ${account.name} is missing ${missingOrEmpty.length} variables: ${missingOrEmpty.join(', ')}`);

  // Build merged payload preserving existing keys and supplementing missing keys
  const mergedPayload = [];
  for (const [key, value] of existingMap.entries()) {
    const finalVal = (value !== '' && value !== null && value !== undefined) ? value : (masterConfig[key] || '');
    mergedPayload.push({ key, value: finalVal });
  }

  for (const key of missingOrEmpty) {
    if (!existingMap.has(key)) {
      mergedPayload.push({ key, value: masterConfig[key] || '' });
    }
  }

  try {
    const putRes = await fetch(`https://api.render.com/v1/services/${account.serviceId}/env-vars`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(mergedPayload)
    });

    if (putRes.ok) {
      console.log(`[Failover] ✅ Injected ${missingOrEmpty.length} missing environment variables into ${account.name}!`);
      return { synced: true, updated: true, missingCount: missingOrEmpty.length, injected: missingOrEmpty };
    } else {
      const errText = await putRes.text();
      console.error(`[Failover] ❌ Failed injecting env vars into ${account.name}: HTTP ${putRes.status} - ${errText}`);
      return { synced: false, updated: false, reason: `put_error_${putRes.status}` };
    }
  } catch (err) {
    console.error(`[Failover] Error pushing env vars to ${account.name}:`, err.message);
    return { synced: false, updated: false, reason: err.message };
  }
}

async function getRecentDeploys(account, limit = 5) {
  if (!account.apiKey || !account.serviceId) return [];
  try {
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}/deploys?limit=${limit}`, {
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json'
      }
    });
    if (res.ok) {
      const data = await res.json();
      return Array.isArray(data) ? data.map(item => item.deploy || item) : [];
    }
  } catch (err) {
    console.warn(`[Failover] Failed fetching deploys for ${account.name}: ${err.message}`);
  }
  return [];
}

function hasRecentDeployWithinMinutes(deploys, minutes = 30) {
  if (!Array.isArray(deploys) || deploys.length === 0) return false;
  const cutoff = Date.now() - (minutes * 60 * 1000);
  for (const dep of deploys) {
    const timestampStr = dep.createdAt || dep.startedAt;
    if (timestampStr) {
      const t = new Date(timestampStr).getTime();
      if (!isNaN(t) && t > cutoff) {
        return true;
      }
    }
  }
  return false;
}

async function getServiceStatus(account) {
  if (!account.apiKey) {
    console.warn(`[Failover] Warning: No API key provided for ${account.name}.`);
    return null;
  }

  // 1. Direct fetch by service ID if present
  if (account.serviceId) {
    try {
      const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}`, {
        headers: {
          'Authorization': `Bearer ${account.apiKey}`,
          'Accept': 'application/json'
        }
      });
      if (res.ok) {
        const data = await res.json();
        if (data.serviceDetails?.url) {
          account.url = data.serviceDetails.url;
        }
        return data;
      }
    } catch (err) {
      console.warn(`[Failover] Direct service check failed for ${account.name} (${account.serviceId}): ${err.message}. Trying dynamic discovery...`);
    }
  }

  // 2. Dynamic discovery: list services on this Render account to locate the bot service
  try {
    const res = await fetch(`https://api.render.com/v1/services?limit=20`, {
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json'
      }
    });
    if (res.ok) {
      const services = await res.json();
      const match = services.find(item => {
        const s = item.service || item;
        return (s.name && s.name.toLowerCase().includes('whatsapp-patient-bot')) ||
               (s.serviceDetails?.url && s.serviceDetails.url.toLowerCase().includes('whatsapp-patient-bot'));
      }) || services[0];

      if (match) {
        const s = match.service || match;
        account.serviceId = s.id;
        if (s.serviceDetails?.url) {
          account.url = s.serviceDetails.url;
        }
        console.log(`[Failover] 🔎 Dynamically discovered service for ${account.name}: ID=${account.serviceId}, URL=${account.url}`);
        return s;
      }
    }
  } catch (err) {
    console.error(`[Failover] Dynamic service resolution error for ${account.name}:`, err.message);
  }

  return null;
}

async function resumeService(account) {
  if (!account.apiKey || !account.serviceId) return false;
  try {
    console.log(`[Failover] Resuming service ${account.serviceId} on ${account.name}...`);
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}/resume`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json'
      }
    });
    if (res.ok) {
      console.log(`[Failover] ✅ Resume accepted for ${account.name}`);
      return true;
    }
    const errText = await res.text();
    console.warn(`[Failover] Resume failed for ${account.name}: HTTP ${res.status} - ${errText}`);
    return false;
  } catch (err) {
    console.error(`[Failover] Error resuming ${account.name}:`, err.message);
    return false;
  }
}

async function suspendService(account) {
  if (!account.apiKey || !account.serviceId) return false;
  try {
    console.log(`[Failover] Putting standby service ${account.serviceId} on ${account.name} to sleep (suspend)...`);
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}/suspend`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json'
      }
    });
    return res.ok;
  } catch (err) {
    console.error(`[Failover] Error suspending ${account.name}:`, err.message);
    return false;
  }
}

async function triggerDeploy(account) {
  if (!account.apiKey || !account.serviceId) return false;
  try {
    console.log(`[Failover] 🚀 Triggering deployment on ${account.name} (${account.serviceId}) to ensure latest code...`);
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}/deploys`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ clearCache: 'do_not_clear' })
    });
    if (res.ok) {
      const data = await res.json();
      console.log(`[Failover] ✅ Deploy triggered successfully: ID=${data.id}, status=${data.status}`);
      return true;
    }
    const errText = await res.text();
    console.error(`[Failover] Failed to trigger deploy on ${account.name}: HTTP ${res.status} - ${errText}`);
    return false;
  } catch (err) {
    console.error(`[Failover] Error triggering deploy on ${account.name}:`, err.message);
    return false;
  }
}

async function updateCronJobUrl(newUrl) {
  if (!CRON_JOB_API_KEY) {
    console.warn(`[Failover] Warning: CRON_JOB_API_KEY is not set. Cannot update cron-job.org.`);
    return false;
  }
  try {
    const targetPingUrl = newUrl.endsWith('/ping') ? newUrl : `${newUrl.replace(/\/$/, '')}/ping`;
    console.log(`[Failover] Updating cron-job.org (${CRON_JOB_ID}) to: ${targetPingUrl}`);
    const res = await fetch(`https://api.cron-job.org/jobs/${CRON_JOB_ID}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${CRON_JOB_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        job: {
          url: targetPingUrl,
          saveResponses: false,
          enabled: true,
          requestTimeout: 60
        }
      })
    });
    if (res.ok) {
      console.log(`[Failover] 🔄 cron-job.org updated successfully to ${targetPingUrl} (timeout: 60s)`);
      return true;
    }
    const errText = await res.text();
    console.warn(`[Failover] Failed to update cron-job.org: HTTP ${res.status} - ${errText}`);
    return false;
  } catch (err) {
    console.error(`[Failover] Error updating cron-job.org:`, err.message);
    return false;
  }
}

async function updateCronJobWatchdog(githubToken) {
  if (!CRON_JOB_API_KEY || !githubToken) return false;
  try {
    const res = await fetch(`https://api.cron-job.org/jobs/${CRON_JOB_WATCHDOG_ID}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${CRON_JOB_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        job: {
          extendedData: {
            headers: {
              'Accept': 'application/vnd.github.v3+json',
              'Authorization': `Bearer ${githubToken}`,
              'User-Agent': 'cron-job.org-watchdog'
            },
            body: JSON.stringify({ ref: 'main' })
          }
        }
      })
    });
    if (res.ok) {
      console.log(`[Failover] 🔄 Watchdog cron-job (${CRON_JOB_WATCHDOG_ID}) token verified & refreshed.`);
      return true;
    }
  } catch (_) {}
  return false;
}

async function checkAccountHealth(account) {
  if (!account.url) return { ok: false, reason: 'no_url' };
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 35000);
    const res = await fetch(`${account.url}/health`, { signal: controller.signal });
    clearTimeout(timeout);

    if (!res.ok) {
      return { ok: false, status: res.status, reason: `HTTP_${res.status}` };
    }

    let json = await res.json();
    console.log(`[Failover] 🩺 HTTP Health for ${account.name}: connected=${json.connected}, uptime=${Math.round(json.uptime || 0)}s, botUser=${json.botUser?.id || 'none'}`);

    // Fresh cold start grace period (Baileys socket handshake takes 20-45s)
    if (!json.connected && (json.uptime || 0) < 90) {
      console.log(`[Failover] ⏳ Service ${account.name} recently booted (uptime: ${Math.round(json.uptime || 0)}s). Waiting 25s for initial socket handshake...`);
      await new Promise(r => setTimeout(r, 25000));
      try {
        const bootRes = await fetch(`${account.url}/health`);
        if (bootRes.ok) {
          json = await bootRes.json();
          console.log(`[Failover] 🩺 Fresh boot re-check: connected=${json.connected}, uptime=${Math.round(json.uptime || 0)}s`);
        }
      } catch (_) {}
    }

    // If temporarily disconnected on an established server, wait 12s and re-check before escalating
    if (!json.connected && (json.uptime || 0) >= 90) {
      console.log(`[Failover] ⏳ Socket not connected on ${account.name}, waiting 12s for routine reconnect...`);
      await new Promise(r => setTimeout(r, 12000));
      try {
        const retryRes = await fetch(`${account.url}/health`);
        if (retryRes.ok) {
          json = await retryRes.json();
          console.log(`[Failover] 🩺 Re-check result: connected=${json.connected}`);
        }
      } catch (_) {}
    }

    // If STILL not connected after grace period, call in-memory /reconnect endpoint
    if (!json.connected && (json.uptime || 0) >= 90) {
      console.warn(`[Failover] 🔄 Socket still disconnected. Requesting in-memory /reconnect on ${account.name}...`);
      try {
        const recRes = await fetch(`${account.url}/reconnect`);
        if (recRes.ok) {
          const recJson = await recRes.json();
          console.log(`[Failover] In-memory reconnect result:`, recJson);
        }
        // Wait 10s and re-verify
        await new Promise(r => setTimeout(r, 10000));
        const finalRes = await fetch(`${account.url}/health`);
        if (finalRes.ok) {
          json = await finalRes.json();
        }
      } catch (rErr) {
        console.error(`[Failover] Failed calling /reconnect:`, rErr.message);
      }
    }

    // Check if outdated code
    if (json.telegramConfigured === undefined) {
      const recentDeploys = await getRecentDeploys(account, 3);
      if (!hasRecentDeployWithinMinutes(recentDeploys, 30)) {
        console.warn(`[Failover] ⚠️ ${account.name} is running older code missing modern features. Triggering deploy...`);
        await triggerDeploy(account);
      }
    }

    return {
      ok: json.connected === true,
      data: json,
      reason: json.connected ? 'healthy' : 'socket_disconnected'
    };
  } catch (err) {
    return { ok: false, reason: err.message };
  }
}

async function checkAndFailover() {
  console.log(`\n=============================================================`);
  console.log(`[Failover] 🔍 Scanning Render Accounts & Services (${new Date().toISOString()})`);
  console.log(`=============================================================`);

  const masterConfig = await resolveMasterEnvConfig();
  const accountsStatus = [];

  for (const acc of RENDER_ACCOUNTS) {
    const srv = await getServiceStatus(acc);
    if (!srv) {
      accountsStatus.push({ account: acc, state: 'UNREACHABLE', suspenders: [], isSuspended: true, isBillingSuspended: false });
      continue;
    }

    const isSuspended = srv.suspended === 'suspended';
    const suspenders = Array.isArray(srv.suspenders) ? srv.suspenders : [];
    const isBillingSuspended = isSuspended && suspenders.includes('billing');

    let state = 'STANDBY';
    if (!isSuspended) {
      state = 'ACTIVE';
    } else if (isBillingSuspended) {
      state = 'BILLING_EXHAUSTED';
    }

    console.log(`[Failover] ${acc.name} (${acc.serviceId}): state=${state}, suspended=${srv.suspended}, suspenders=${JSON.stringify(suspenders)}`);

    accountsStatus.push({
      account: acc,
      service: srv,
      state,
      isSuspended,
      isBillingSuspended,
      suspenders
    });
  }

  // Enforce single-container multi-device mutual exclusivity
  const activeEntries = accountsStatus.filter(e => e.state === 'ACTIVE');
  let activeEntry = null;
  let activeHealthy = false;

  if (activeEntries.length > 1) {
    console.warn(`\n[Failover] ⚠️ CONFLICT HAZARD: Detected ${activeEntries.length} simultaneous active services!`);
    console.warn(`[Failover] Checking which active service is healthy to eliminate multi-container stream conflicts...`);
    
    // Check health of each active service
    for (const entry of activeEntries) {
      const health = await checkAccountHealth(entry.account);
      entry.uptime = health.data?.uptime || 0;
      if (health.ok && !activeEntry) {
        activeEntry = entry;
        activeHealthy = true;
        console.log(`[Failover] 🎯 Preserving primary healthy service: ${entry.account.name}`);
      } else {
        console.log(`[Failover] 🛑 Immediately suspending competing active service: ${entry.account.name}...`);
        await suspendService(entry.account);
        entry.state = 'STANDBY';
        entry.isSuspended = true;
      }
    }
  } else if (activeEntries.length === 1) {
    activeEntry = activeEntries[0];
    console.log(`\n[Failover] 🟢 Found ACTIVE service: ${activeEntry.account.name} (${activeEntry.account.url})`);
    const health = await checkAccountHealth(activeEntry.account);
    activeEntry.uptime = health.data?.uptime || 0;
    if (health.ok) {
      activeHealthy = true;
      console.log(`[Failover] ✅ Active service ${activeEntry.account.name} is 100% HEALTHY.`);
    } else {
      console.warn(`[Failover] ⚠️ Active service ${activeEntry.account.name} failed health check: ${health.reason}`);
    }
  }

  // Case 1: Active service is verified healthy
  if (activeHealthy && activeEntry) {
    await updateCronJobUrl(activeEntry.account.url);
    await updateCronJobWatchdog(GITHUB_TOKEN);

    // Verify active service has complete configuration
    await syncAccountEnvVars(activeEntry.account, masterConfig);

    // Put any awake standby services to sleep to conserve free hours
    for (const entry of accountsStatus) {
      if (entry.account.serviceId !== activeEntry.account.serviceId && !entry.isSuspended) {
        console.log(`[Failover] 💤 Standby account ${entry.account.name} was awake. Suspending to conserve free hours...`);
        await suspendService(entry.account);
      }
    }

    // -------------------------------------------------------------------------------------------------
    // Autonomous Quota Budgeter & Planned 10-Day Rotation Policy
    // Ensures each account runs for at least 10 days, then gracefully rotates to preserve ~510h in reserve.
    // -------------------------------------------------------------------------------------------------
    const tourDays = await getActiveTourDays(activeEntry.account, activeEntry.service, activeEntry.uptime);
    console.log(`[Failover] ⏱️ Tour status: ${activeEntry.account.name} has served ${tourDays.toFixed(2)} days (Policy threshold: ${TOUR_ROTATION_DAYS} days).`);

    const eligibleStandbys = accountsStatus.filter(e => 
      !e.isBillingSuspended && 
      e.state !== 'UNREACHABLE' &&
      e.account.serviceId !== activeEntry.account.serviceId
    );

    if (tourDays >= TOUR_ROTATION_DAYS && eligibleStandbys.length > 0) {
      console.log(`\n[Failover] 🔄 PLANNED ${TOUR_ROTATION_DAYS}-DAY ROTATION TRIGGERED!`);
      console.log(`[Failover] Active service ${activeEntry.account.name} has completed its planned ${tourDays.toFixed(1)}-day cycle.`);
      console.log(`[Failover] Rotating to next healthy standby to preserve remaining free hours on ${activeEntry.account.name}...`);

      const nextTarget = eligibleStandbys[0];
      console.log(`[Failover] 🎯 Rotating active duty to: ${nextTarget.account.name} (${nextTarget.account.serviceId})`);

      // 1. Sync all environment variables on target candidate before activation
      await syncAccountEnvVars(nextTarget.account, masterConfig);

      // 2. Suspend old active instance first to avoid multi-device stream conflicts (Code: 440)
      console.log(`[Failover] 💤 Suspending current active service ${activeEntry.account.name}...`);
      await suspendService(activeEntry.account);

      // 3. Resume next target service
      console.log(`[Failover] 🚀 Resuming next candidate ${nextTarget.account.name}...`);
      const resumed = await resumeService(nextTarget.account);
      if (resumed) {
        // Record fresh tour start timestamp on next target
        await setAccountTourStart(nextTarget.account, new Date().toISOString());

        // Trigger deploy on new target
        await triggerDeploy(nextTarget.account);

        // Update cron-job.org keep-alive ping URL to new target
        await updateCronJobUrl(nextTarget.account.url);
        await updateCronJobWatchdog(GITHUB_TOKEN);

        console.log(`[Failover] 🎉 PLANNED ROTATION COMPLETED! Activated ${nextTarget.account.name} at ${nextTarget.account.url}.\n`);
        return { success: true, rotated: true, from: activeEntry.account.name, to: nextTarget.account.name };
      } else {
        console.error(`[Failover] ❌ Target resume failed on ${nextTarget.account.name}. Re-activating fallback ${activeEntry.account.name}...`);
        await resumeService(activeEntry.account);
      }
    } else if (tourDays >= TOUR_ROTATION_DAYS && eligibleStandbys.length === 0) {
      console.log(`[Failover] ℹ️ Tour threshold reached (${tourDays.toFixed(1)} days), but no standby candidates are currently available (others billing-suspended or waiting for month reset). Continuing on ${activeEntry.account.name}.`);
    }

    console.log(`[Failover] ✅ Failover scan complete. System stable on ${activeEntry.account.name}.\n`);
    return { success: true, active: activeEntry.account.name, tourDays };
  }

  // Case 2: Active service is UNHEALTHY or ALL accounts are suspended -> INITIATE FAILOVER
  console.log(`\n[Failover] 🚨 Failover REQUIRED! Searching for eligible healthy standby candidate...`);

  // Filter candidates: must not be billing-exhausted, must not be unreachable, and different serviceId from unhealthy active
  const candidates = accountsStatus.filter(e => 
    !e.isBillingSuspended && 
    e.state !== 'UNREACHABLE' &&
    (!activeEntry || e.account.serviceId !== activeEntry.account.serviceId)
  );

  if (candidates.length === 0) {
    console.warn(`[Failover] ℹ️ Notice: No other standby accounts available without billing exhaustion.`);
    if (activeEntry) {
      const currentUptime = activeEntry.uptime || 0;

      // 1. Self-healing: verify and inject any missing environment variables first
      console.log(`[Failover] 🔍 Verifying configuration integrity on ${activeEntry.account.name}...`);
      const syncRes = await syncAccountEnvVars(activeEntry.account, masterConfig);
      if (syncRes.updated) {
        console.log(`[Failover] 🚀 Missing configuration injected into ${activeEntry.account.name}. Triggering fresh deployment to apply...`);
        await triggerDeploy(activeEntry.account);
        await updateCronJobUrl(activeEntry.account.url);
        return { success: true, healed: 'env_vars_injected', active: activeEntry.account.name };
      }

      // 2. Anti-thrashing guard: verify MONGODB_URI is present and check deploy frequency
      const envMap = await getAccountEnvVars(activeEntry.account);
      if (!envMap || !envMap.get('MONGODB_URI')) {
        console.error(`[Failover] 🛑 Breaking redeploy loop: Database credentials (MONGODB_URI) are missing on ${activeEntry.account.name}. Redeploying would thrash without healing.`);
        await updateCronJobUrl(activeEntry.account.url);
        return { success: false, error: 'missing_database_credentials' };
      }

      const recentDeploys = await getRecentDeploys(activeEntry.account, 3);
      const recentlyDeployed = hasRecentDeployWithinMinutes(recentDeploys, 30);

      if (recentlyDeployed) {
        console.warn(`[Failover] 🛑 Breaking redeploy loop: Service ${activeEntry.account.name} was redeployed within the last 30 minutes. Allowing socket/database to stabilize without redeploy thrashing.`);
      } else if (currentUptime >= 180) {
        console.warn(`[Failover] Active service has persistent disconnect (uptime: ${Math.round(currentUptime)}s) and no recent redeploy. Triggering emergency redeploy on ${activeEntry.account.name}...`);
        await triggerDeploy(activeEntry.account);
      } else {
        console.log(`[Failover] Service recently deployed or booting (uptime: ${Math.round(currentUptime)}s). Giving socket time to stabilize on ${activeEntry.account.name}.`);
      }
      await updateCronJobUrl(activeEntry.account.url);
    }
    return { success: false, error: 'all_other_accounts_exhausted' };
  }

  // Pick first available candidate
  const target = candidates[0];
  console.log(`[Failover] 🎯 Selected failover target: ${target.account.name} (${target.account.serviceId})`);

  // Ensure environment variables are synchronized before activating or resuming target standby
  console.log(`[Failover] 🔄 Synchronizing environment variables for standby candidate ${target.account.name}...`);
  await syncAccountEnvVars(target.account, masterConfig);

  // If old active service was unhealthy, put it to sleep to avoid socket conflicts
  if (activeEntry && activeEntry.account.serviceId !== target.account.serviceId) {
    console.log(`[Failover] Suspending failing service ${activeEntry.account.name}...`);
    await suspendService(activeEntry.account);
  }

  // Resume target service
  const resumed = await resumeService(target.account);
  if (!resumed) {
    console.error(`[Failover] ❌ Failed to resume ${target.account.name}! Trying next available...`);
    // Try remaining candidates if any
    for (let c = 1; c < candidates.length; c++) {
      const alt = candidates[c];
      console.log(`[Failover] Attempting resume on alternate: ${alt.account.name}...`);
      await syncAccountEnvVars(alt.account, masterConfig);
      if (await resumeService(alt.account)) {
        await setAccountTourStart(alt.account, new Date().toISOString());
        await triggerDeploy(alt.account);
        await updateCronJobUrl(alt.account.url);
        await updateCronJobWatchdog(GITHUB_TOKEN);
        return { success: true, active: alt.account.name };
      }
    }
    return { success: false, error: 'resume_failed' };
  }

  // Record fresh tour start timestamp on target
  await setAccountTourStart(target.account, new Date().toISOString());

  // Trigger fresh deployment to guarantee latest code is running
  await triggerDeploy(target.account);

  // Update cron-job.org
  await updateCronJobUrl(target.account.url);
  await updateCronJobWatchdog(GITHUB_TOKEN);

  // Suspend other awake standbys
  for (const entry of accountsStatus) {
    if (entry.account.serviceId !== target.account.serviceId && !entry.isSuspended && !entry.isBillingSuspended) {
      await suspendService(entry.account);
    }
  }

  console.log(`[Failover] 🎉 FAILOVER SUCCESSFUL! Activated ${target.account.name} at ${target.account.url}\n`);
  return { success: true, active: target.account.name };
}

// Auto-run if executed as script
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  checkAndFailover().catch(err => {
    console.error('[Failover] Fatal execution error:', err);
    process.exit(1);
  });
}

export {
  checkAndFailover,
  RENDER_ACCOUNTS,
  updateCronJobUrl,
  updateCronJobWatchdog,
  MASTER_ENV_CONFIG,
  resolveMasterEnvConfig,
  getAccountEnvVars,
  syncAccountEnvVars,
  getRecentDeploys,
  hasRecentDeployWithinMinutes,
  TOUR_ROTATION_DAYS,
  getActiveTourDays,
  setAccountTourStart
};
