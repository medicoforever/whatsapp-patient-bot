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

const RENDER_ACCOUNTS = [
  {
    name: 'medicoforever003',
    apiKey: (process.env.RENDER_API_KEY_003 || process.env.RENDER_API_KEY_4 || process.env.RENDER_API_KEY_2 || '').trim() || 'rnd_k7763fBzTQHt7R0crVsSQrWYoKFB',
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
    apiKey: (process.env.RENDER_API_KEY_002 || process.env.RENDER_API_KEY_2 || process.env.RENDER_API_KEY_3 || '').trim() || 'rnd_jYutYuSK6dtAZwYKFbrjI1ekhffB',
    serviceId: (process.env.RENDER_SERVICE_ID_002 || process.env.RENDER_SERVICE_ID_2 || '').trim() || 'srv-d5jatbq4d50c73fpbgcg',
    url: 'https://whatsapp-patient-bot.onrender.com'
  },
  {
    name: 'raddoc1996',
    apiKey: (process.env.RENDER_API_KEY_RADDOC || process.env.RENDER_API_KEY_3 || process.env.RENDER_API_KEY_4 || '').trim() || 'rnd_ATJs45AaaYcnkL3SETD3vBdkWVmf',
    serviceId: (process.env.RENDER_SERVICE_ID_RADDOC || process.env.RENDER_SERVICE_ID_3 || '').trim() || 'srv-d9uvkuvavr4c73bljb10',
    url: 'https://whatsapp-patient-bot-b4tl.onrender.com'
  },
  {
    name: 'medicoforever003_pool3',
    apiKey: (process.env.RENDER_API_KEY_003_POOL3 || process.env.RENDER_API_KEY_5 || '').trim() || 'rnd_Ma8PeY6Nkq81TNZeozNkixIJg7eC',
    serviceId: (process.env.RENDER_SERVICE_ID_5 || '').trim() || 'srv-da46e0fqj5pc73bdboqg',
    url: 'https://whatsapp-patient-bot-zcmf.onrender.com'
  }
];

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

    // If temporarily disconnected, wait 12s and re-check before escalating
    if (!json.connected) {
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
    if (!json.connected) {
      console.warn(`[Failover] 🔄 Socket still disconnected. Requesting in-memory /reconnect on ${account.name}...`);
      try {
        const recRes = await fetch(`${account.url}/reconnect`);
        const recJson = await recRes.json();
        console.log(`[Failover] In-memory reconnect result:`, recJson);
        // Wait 8s and re-verify
        await new Promise(r => setTimeout(r, 8000));
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
      console.warn(`[Failover] ⚠️ ${account.name} is running older code missing modern features. Triggering deploy...`);
      await triggerDeploy(account);
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

  // Find currently active (not_suspended) service
  let activeEntry = accountsStatus.find(e => e.state === 'ACTIVE');
  let activeHealthy = false;

  if (activeEntry) {
    console.log(`\n[Failover] 🟢 Found ACTIVE service: ${activeEntry.account.name} (${activeEntry.account.url})`);
    const health = await checkAccountHealth(activeEntry.account);
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

    // Put any awake standby services to sleep to conserve free hours
    for (const entry of accountsStatus) {
      if (entry.account.serviceId !== activeEntry.account.serviceId && !entry.isSuspended) {
        console.log(`[Failover] 💤 Standby account ${entry.account.name} was awake. Suspending to conserve free hours...`);
        await suspendService(entry.account);
      }
    }
    console.log(`[Failover] ✅ Failover scan complete. System stable on ${activeEntry.account.name}.\n`);
    return { success: true, active: activeEntry.account.name };
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
    console.error(`[Failover] ❌ CRITICAL: No eligible standby accounts found without billing exhaustion!`);
    if (activeEntry) {
      console.warn(`[Failover] Attempting emergency redeploy on current active ${activeEntry.account.name}...`);
      await triggerDeploy(activeEntry.account);
      await updateCronJobUrl(activeEntry.account.url);
    }
    return { success: false, error: 'all_accounts_exhausted' };
  }

  // Pick first available candidate
  const target = candidates[0];
  console.log(`[Failover] 🎯 Selected failover target: ${target.account.name} (${target.account.serviceId})`);

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
      if (await resumeService(alt.account)) {
        await triggerDeploy(alt.account);
        await updateCronJobUrl(alt.account.url);
        await updateCronJobWatchdog(GITHUB_TOKEN);
        return { success: true, active: alt.account.name };
      }
    }
    return { success: false, error: 'resume_failed' };
  }

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

export { checkAndFailover, RENDER_ACCOUNTS, updateCronJobUrl, updateCronJobWatchdog };
