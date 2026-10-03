/**

 * 24/7 Render Failover & Uptime Manager
 * 
 * Logic:
 * 1. Checks all 4 Render accounts sequentially in priority order.
 * 2. Checks active service HTTP /health.
 * 3. If socket is temporarily reconnecting, waits 12s and re-checks, then calls /reconnect if needed (NEVER triggers cold redeploys for routine reconnects!).
 * 4. Puts standby accounts to sleep (suspend) to preserve their 750 free hours.
 * 5. If the active account exhausts hours (billing suspension), automatically awakens the next standby account in line.
 * 6. Keeps cron-job.org pointing to the active service.
 */

const CRON_JOB_API_KEY = process.env.CRON_JOB_API_KEY || '3GKdFNCZXErgSKSCeHMXG2SGrBYzGN6pldcaHqBHdb8=';
const CRON_JOB_ID = process.env.CRON_JOB_ID || '7156467';

const RENDER_ACCOUNTS = [
  {
    name: 'medicoforever008',
    apiKey: process.env.RENDER_API_KEY_1 || process.env.RENDER_API_KEY_008 || 'rnd_NNtHMoAwbdGttv0F4WDvFqnbg8bR',
    serviceId: process.env.RENDER_SERVICE_ID_1 || 'srv-daa4ugpf2nfc739834v0',
    url: 'https://whatsapp-patient-bot-f9lc.onrender.com'
  },
  {
    name: 'medicoforever002',
    apiKey: process.env.RENDER_API_KEY_2 || process.env.RENDER_API_KEY_002 || 'rnd_jYutYuSK6dtAZwYKFbrjI1ekhffB',
    serviceId: process.env.RENDER_SERVICE_ID_2 || 'srv-d5jatbq4d50c73fpbgcg',
    url: 'https://whatsapp-patient-bot.onrender.com'
  },
  {
    name: 'raddoc1996',
    apiKey: process.env.RENDER_API_KEY_3 || process.env.RENDER_API_KEY_RADDOC || 'rnd_ATJs45AaaYcnkL3SETD3vBdkWVmf',
    serviceId: process.env.RENDER_SERVICE_ID_3 || 'srv-d9uvkuvavr4c73bljb10',
    url: 'https://whatsapp-patient-bot-b4tl.onrender.com'
  },
  {
    name: 'medicoforever003',
    apiKey: process.env.RENDER_API_KEY_4 || process.env.RENDER_API_KEY_003 || 'rnd_k7763fBzTQHt7R0crVsSQrWYoKFB',
    serviceId: process.env.RENDER_SERVICE_ID_4 || 'srv-da46e0fqj5pc73bdboqg',
    url: 'https://whatsapp-patient-bot-zcmf.onrender.com'
  }
];

async function getServiceStatus(account) {
  if (!account.apiKey) {
    console.warn(`[Failover] Warning: No API key provided for ${account.name}. Check environment variables/secrets.`);
    return null;
  }
  try {
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}`, {
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json'
      }
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error(`[Failover] Error checking ${account.name}:`, err.message);
    return null;
  }
}

async function resumeService(account) {
  if (!account.apiKey) return false;
  try {
    console.log(`[Failover] Resuming service ${account.serviceId} on ${account.name}...`);
    const res = await fetch(`https://api.render.com/v1/services/${account.serviceId}/resume`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${account.apiKey}`,
        'Accept': 'application/json'
      }
    });
    return res.ok;
  } catch (err) {
    console.error(`[Failover] Error resuming ${account.name}:`, err.message);
    return false;
  }
}

async function suspendService(account) {
  if (!account.apiKey) return false;
  try {
    console.log(`[Failover] Putting standby service ${account.serviceId} on ${account.name} into suspended state...`);
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
  if (!account.apiKey) return false;
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
    console.log(`[Failover] Updating cron-job.org to: ${targetPingUrl}`);
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
    throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    console.error(`[Failover] Error updating cron-job.org:`, err.message);
    return false;
  }
}

async function checkAndFailover() {
  console.log(`\n[Failover] 🔍 Checking Render accounts status... (${new Date().toISOString()})`);

  let activeAccount = null;

  for (const acc of RENDER_ACCOUNTS) {
    const srv = await getServiceStatus(acc);
    if (!srv) continue;

    const isSuspended = srv.suspended === 'suspended';
    const isBillingSuspended = isSuspended && Array.isArray(srv.suspenders) && srv.suspenders.includes('billing');

    console.log(`[Failover] ${acc.name}: ${srv.suspended} (suspenders: ${JSON.stringify(srv.suspenders || [])})`);

    if (!isSuspended) {
      activeAccount = acc;
      console.log(`[Failover] ✅ Currently ACTIVE account: ${acc.name} (${acc.url})`);
      break;
    }

    if (isBillingSuspended) {
      console.log(`[Failover] ⚠️ Account ${acc.name} is suspended for BANDWIDTH/BILLING. Checking next...`);
    }
  }

  // If the active account is found, verify its HTTP health & make sure cron-job.org points to it
  if (activeAccount) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 45000);
      let hRes = await fetch(`${activeAccount.url}/health`, { signal: controller.signal });
      clearTimeout(timeout);

      if (hRes.ok) {
        let hJson = await hRes.json();
        console.log(`[Failover] 🩺 HTTP Health OK: connected=${hJson.connected}, uptime=${Math.round(hJson.uptime)}s, botUser=${hJson.botUser?.id || 'none'}`);

        // If temporarily disconnected (routine 5s reconnect), wait 12s and re-check before jumping to conclusions
        if (!hJson.connected) {
          console.log(`[Failover] ⏳ Socket not connected, waiting 12s to see if routine reconnect is finishing...`);
          await new Promise(r => setTimeout(r, 12000));
          try {
            const retryRes = await fetch(`${activeAccount.url}/health`);
            if (retryRes.ok) {
              hJson = await retryRes.json();
              console.log(`[Failover] 🩺 Re-check result: connected=${hJson.connected}`);
            }
          } catch (_) {}
        }

        // If STILL not connected after waiting, call /reconnect endpoint to reset socket in-memory without a cold redeploy
        if (!hJson.connected) {
          console.warn(`[Failover] 🔄 Socket still disconnected. Calling in-memory /reconnect endpoint...`);
          try {
            const recRes = await fetch(`${activeAccount.url}/reconnect`);
            const recJson = await recRes.json();
            console.log(`[Failover] In-memory reconnect requested:`, recJson);
          } catch (rErr) {
            console.error(`[Failover] Failed calling /reconnect:`, rErr.message);
          }
        }

        // Detect if active service is running outdated code missing critical features
        if (hJson.telegramConfigured === undefined) {
          console.warn(`[Failover] ⚠️ Active account ${activeAccount.name} is running outdated code missing modern features. Triggering deploy...`);
          await triggerDeploy(activeAccount);
        }
      } else {
        console.warn(`[Failover] ⚠️ Health check returned HTTP ${hRes.status}.`);
      }
    } catch (err) {
      console.warn(`[Failover] ⚠️ Health check request error: ${err.message}.`);
    }

    await updateCronJobUrl(activeAccount.url);
    console.log(`[Failover] System status checked on ${activeAccount.name}`);

    // If month reset un-suspended any standby accounts, put them back to sleep
    for (const acc of RENDER_ACCOUNTS) {
      if (acc.serviceId !== activeAccount.serviceId) {
        const srv = await getServiceStatus(acc);
        if (srv && srv.suspended !== 'suspended') {
          console.log(`[Failover] 💤 Standby account ${acc.name} was un-suspended (e.g. month-reset). Suspending to preserve hours...`);
          await suspendService(acc);
        }
      }
    }
    return;
  }

  // If all are suspended, attempt to resume the next available one (e.g. at month reset)
  console.log(`[Failover] ⚠️ All accounts suspended. Attempting sequential resume...`);
  for (const acc of RENDER_ACCOUNTS) {
    const resumed = await resumeService(acc);
    if (resumed) {
      // Trigger a deploy to guarantee latest code is running on this newly activated service
      await triggerDeploy(acc);
      await updateCronJobUrl(acc.url);
      console.log(`[Failover] ✅ Successfully switched to ${acc.name} with fresh deploy triggered`);
      break;
    }
  }
}

// Execute check
checkAndFailover();

export { checkAndFailover, RENDER_ACCOUNTS, updateCronJobUrl };
