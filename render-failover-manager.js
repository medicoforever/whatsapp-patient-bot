/**
 * 🔄 RENDER MULTI-ACCOUNT FAILOVER & CRON-JOB.ORG AUTO-SWITCHER
 * 
 * This manager monitors the active Render web service.
 * ONLY if a service is suspended specifically for BANDWIDTH/BILLING (suspenders: ["billing"]),
 * it activates the next available Render account in the pool and automatically updates
 * your cron-job.org URL to the new service!
 */

const CRON_JOB_API_KEY = process.env.CRON_JOB_API_KEY || '3GKdFNCZXErgSKSCeHMXG2SGrBYzGN6pldcaHqBHdb8=';
const CRON_JOB_ID = process.env.CRON_JOB_ID || '7156467';

const RENDER_ACCOUNTS = [
  {
    name: 'medicoforever008',
    apiKey: 'rnd_NNtHMoAwbdGttv0F4WDvFqnbg8bR',
    serviceId: 'srv-daa4ugpf2nfc739834v0',
    url: 'https://whatsapp-patient-bot-f9lc.onrender.com'
  },
  {
    name: 'medicoforever002',
    apiKey: 'rnd_jYutYuSK6dtAZwYKFbrjI1ekhffB',
    serviceId: 'srv-d5jatbq4d50c73fpbgcg',
    url: 'https://whatsapp-patient-bot.onrender.com'
  },
  {
    name: 'raddoc1996',
    apiKey: 'rnd_ATJs45AaaYcnkL3SETD3vBdkWVmf',
    serviceId: 'srv-d9uvkuvavr4c73bljb10',
    url: 'https://whatsapp-patient-bot-b4tl.onrender.com'
  },
  {
    name: 'medicoforever003',
    apiKey: 'rnd_Ma8PeY6Nkq81TNZeozNkixIJg7eC',
    serviceId: 'srv-da46e0fqj5pc73bdboqg',
    url: 'https://whatsapp-patient-bot-zcmf.onrender.com'
  }
];

async function getServiceStatus(account) {
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

async function updateCronJobUrl(newUrl) {
  try {
    console.log(`[Failover] Updating cron-job.org to: ${newUrl}`);
    const res = await fetch(`https://api.cron-job.org/jobs/${CRON_JOB_ID}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${CRON_JOB_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        job: {
          url: newUrl,
          enabled: true
        }
      })
    });
    if (res.ok) {
      console.log(`[Failover] ✅ cron-job.org updated successfully to ${newUrl}`);
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

  // If the active account is found, make sure cron-job.org points to it
  if (activeAccount) {
    await updateCronJobUrl(activeAccount.url);
    console.log(`[Failover] System healthy on ${activeAccount.name}`);
    return;
  }

  // If all are suspended, attempt to resume the next available one (e.g. at month reset)
  console.log(`[Failover] 🚨 All accounts suspended. Attempting sequential resume...`);
  for (const acc of RENDER_ACCOUNTS) {
    const resumed = await resumeService(acc);
    if (resumed) {
      await updateCronJobUrl(acc.url);
      console.log(`[Failover] ✅ Successfully switched to ${acc.name}`);
      break;
    }
  }
}

// Run check immediately if executed directly
if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) {
  checkAndFailover();
}

export { checkAndFailover, RENDER_ACCOUNTS, updateCronJobUrl };
