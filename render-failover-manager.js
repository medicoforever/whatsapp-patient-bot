/**
 * 🔄 RENDER MULTI-ACCOUNT FAILOVER & CRON-JOB.ORG AUTO-SWITCHER
 * 
 * This manager monitors the active Render web service.
 * ONLY if a service is suspended specifically for BANDWIDTH/BILLING (suspenders: ["billing"]),
 * it activates the next available Render account in the pool and automatically updates
 * your cron-job.org URL to the new service!
 */

const CRON_JOB_API_KEY = process.env.CRON_JOB_API_KEY;
const CRON_JOB_ID = process.env.CRON_JOB_ID || '7156467';

const RENDER_ACCOUNTS = [
  {
    name: 'medicoforever008',
    apiKey: process.env.RENDER_API_KEY_1 || process.env.RENDER_API_KEY_008,
    serviceId: process.env.RENDER_SERVICE_ID_1 || 'srv-daa4ugpf2nfc739834v0',
    url: 'https://whatsapp-patient-bot-f9lc.onrender.com'
  },
  {
    name: 'medicoforever002',
    apiKey: process.env.RENDER_API_KEY_2 || process.env.RENDER_API_KEY_002,
    serviceId: process.env.RENDER_SERVICE_ID_2 || 'srv-d5jatbq4d50c73fpbgcg',
    url: 'https://whatsapp-patient-bot.onrender.com'
  },
  {
    name: 'raddoc1996',
    apiKey: process.env.RENDER_API_KEY_3 || process.env.RENDER_API_KEY_RADDOC,
    serviceId: process.env.RENDER_SERVICE_ID_3 || 'srv-d9uvkuvavr4c73bljb10',
    url: 'https://whatsapp-patient-bot-b4tl.onrender.com'
  },
  {
    name: 'medicoforever003',
    apiKey: process.env.RENDER_API_KEY_4 || process.env.RENDER_API_KEY_003,
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
      const hRes = await fetch(`${activeAccount.url}/health`, { signal: controller.signal });
      clearTimeout(timeout);
      if (hRes.ok) {
        const hJson = await hRes.json();
        console.log(`[Failover] 🩺 HTTP Health OK: connected=${hJson.connected}, uptime=${Math.round(hJson.uptime)}s, botUser=${hJson.botUser?.id || 'none'}`);
        if (!hJson.connected || !hJson.telegramConfigured) {
          console.warn(`[Failover] ⚠️ Active service is missing WhatsApp connection or Telegram config. Triggering auto-heal deploy...`);
          await triggerDeploy(activeAccount);
        }
      } else {
        console.warn(`[Failover] ⚠️ Health check returned HTTP ${hRes.status}. Triggering deploy to recover...`);
        await triggerDeploy(activeAccount);
      }
    } catch (err) {
      console.warn(`[Failover] ⚠️ Health check request error: ${err.message}. Triggering deploy to wake up container...`);
      await triggerDeploy(activeAccount);
    }

    await updateCronJobUrl(activeAccount.url);
    console.log(`[Failover] System healthy on ${activeAccount.name}`);

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
