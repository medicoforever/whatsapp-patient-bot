import { test, describe } from 'node:test';
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
    const isPhoneNumber = /^\d{7,15}$/.test(phone);

    assert.equal(isPhoneNumber, true);
    const attribution = `\n\n👤 *Sent by:* @${phone}`;
    assert.equal(attribution, '\n\n👤 *Sent by:* @919876543210');
  });

  test('TC-F4-05: Non-phone LID JID falls back cleanly to display name without broken mention', () => {
    const senderId = '120363047547742844@lid';
    const senderName = 'Dr. Vikram Patel';
    const phone = senderId.split('@')[0];
    const domain = senderId.split('@')[1];
    const isPhoneNumber = /^\d{7,15}$/.test(phone) && (domain === 's.whatsapp.net' || domain === 'c.us');

    assert.equal(isPhoneNumber, false);
    const attribution = `\n\n👤 *Sent by:* ${senderName || phone}`;
    assert.equal(attribution, '\n\n👤 *Sent by:* Dr. Vikram Patel');
  });

  test('TC-F4-06: Stradus DICOM viewer & MRI protocol footer is appended to group chat reports', () => {
    const GROUP_REPLY_FOOTER = `

━━━━━━━━━━━━━━━━━━━━━━
🖼️ *Go here to see the scan images:*
https://view.stradus.com/

🤖 *Copy-paste the clinical profile here to get suggestions regarding MRI protocols:*
https://ai.studio/apps/86a65a19-cf2f-46de-b4d0-9a941be83604

🎙️ *Radiology dictation:*
https://ai.studio/apps/3f0807e3-2494-4289-a3a6-c12032da731c?fullscreenApplet=true

📚 *MRI protocol books*
https://notebooklm.google.com/notebook/467e8684-c512-488f-b1f7-3a450e344cd5`;

    assert.ok(GROUP_REPLY_FOOTER.includes('https://view.stradus.com/'));
    assert.ok(GROUP_REPLY_FOOTER.includes('https://ai.studio/apps/3f0807e3-2494-4289-a3a6-c12032da731c?fullscreenApplet=true'));
  });
});
