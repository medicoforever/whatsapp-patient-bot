// Set environment to test mode before importing index.js
process.env.NODE_ENV = 'test';

import assert from 'assert';
import { unwrapMessage } from '../index.js';

console.log('🧪 Starting Baileys Wrapper Unwrapping Test Suite...\n');

let passed = 0;
let failed = 0;

function runTestCase(id, description, input, expectedType, expectedFieldCheck) {
  try {
    const result = unwrapMessage(input);
    
    if (expectedType === null) {
      assert.strictEqual(result, null, `${id}: Expected null output`);
    } else {
      assert.ok(result, `${id}: Unwrapped result should not be null`);
      const detectedType = Object.keys(result)[0];
      assert.strictEqual(detectedType, expectedType, `${id}: Expected type '${expectedType}', got '${detectedType}'`);
      if (expectedFieldCheck) {
        expectedFieldCheck(result[detectedType]);
      }
    }
    console.log(`  ✅ PASS: [${id}] ${description}`);
    passed++;
  } catch (err) {
    console.error(`  ❌ FAIL: [${id}] ${description}`);
    console.error(`     Error: ${err.message}`);
    failed++;
  }
}

// ----------------------------------------------------------------------
// Test Cases Execution
// ----------------------------------------------------------------------

// TC-W01: Direct Plain Image
runTestCase('TC-W01', 'Direct Plain Image Message (No Wrapper)', {
  imageMessage: { url: 'https://cdn.example.com/chest_xray.jpg', caption: 'Chest X-Ray' }
}, 'imageMessage', (payload) => {
  assert.strictEqual(payload.caption, 'Chest X-Ray');
});

// TC-W02: Single Ephemeral Wrapper
runTestCase('TC-W02', 'Single Ephemeral Message Wrapper', {
  ephemeralMessage: {
    message: {
      imageMessage: { url: 'https://cdn.example.com/ephemeral_xray.jpg', caption: 'Ephemeral X-Ray' }
    }
  }
}, 'imageMessage', (payload) => {
  assert.strictEqual(payload.caption, 'Ephemeral X-Ray');
});

// TC-W03: ViewOnce V3 Wrapper
runTestCase('TC-W03', 'ViewOnce V3 Message Wrapper', {
  viewOnceMessageV3: {
    message: {
      videoMessage: { url: 'https://cdn.example.com/ultrasound.mp4', seconds: 15 }
    }
  }
}, 'videoMessage', (payload) => {
  assert.strictEqual(payload.seconds, 15);
});

// TC-W04: Document With Caption Wrapper
runTestCase('TC-W04', 'Document With Caption Wrapper', {
  documentWithCaptionMessage: {
    message: {
      documentMessage: { url: 'https://cdn.example.com/lab_results.pdf', fileName: 'lab_results.pdf' }
    }
  }
}, 'documentMessage', (payload) => {
  assert.strictEqual(payload.fileName, 'lab_results.pdf');
});

// TC-W05: Album Wrapper
runTestCase('TC-W05', 'Album Message Wrapper', {
  albumMessage: {
    message: {
      imageMessage: { url: 'https://cdn.example.com/album_photo1.jpg', caption: 'Album Photo 1' }
    }
  }
}, 'imageMessage', (payload) => {
  assert.strictEqual(payload.caption, 'Album Photo 1');
});

// TC-W06: Push-To-Video (PTV) Wrapper
runTestCase('TC-W06', 'Push-To-Video (PTV) Message Wrapper', {
  ptvMessage: {
    videoMessage: { url: 'https://cdn.example.com/video_note.mp4', ptv: true }
  }
}, 'videoMessage', (payload) => {
  assert.strictEqual(payload.ptv, true);
});

// TC-W07: Edited Message Wrapper
runTestCase('TC-W07', 'Edited Message Protocol Wrapper', {
  editedMessage: {
    message: {
      protocolMessage: {
        editedMessage: {
          extendedTextMessage: { text: 'Corrected dosage: Paracetamol 500mg TDS' }
        }
      }
    }
  }
}, 'extendedTextMessage', (payload) => {
  assert.strictEqual(payload.text, 'Corrected dosage: Paracetamol 500mg TDS');
});

// TC-W08: Hydrated Template Message Wrapper
runTestCase('TC-W08', 'Hydrated Template Message Wrapper', {
  templateMessage: {
    hydratedTemplate: {
      imageMessage: { url: 'https://cdn.example.com/template_header.jpg', caption: 'Appointment Confirmation' }
    }
  }
}, 'imageMessage', (payload) => {
  assert.strictEqual(payload.caption, 'Appointment Confirmation');
});

// TC-W09: Interactive Message Header Wrapper
runTestCase('TC-W09', 'Interactive Message Header Wrapper', {
  interactiveMessage: {
    header: {
      documentMessage: { url: 'https://cdn.example.com/prescription.pdf', fileName: 'prescription.pdf' }
    },
    body: { text: 'Please review your prescription' }
  }
}, 'documentMessage', (payload) => {
  assert.strictEqual(payload.fileName, 'prescription.pdf');
});

// TC-W10: Triple Nested Wrapper (Ephemeral -> ViewOnceV3 -> DocumentWithCaption)
runTestCase('TC-W10', 'Triple Nested Wrapper (Ephemeral -> ViewOnceV3 -> DocumentWithCaption)', {
  ephemeralMessage: {
    message: {
      viewOnceMessageV3: {
        message: {
          documentWithCaptionMessage: {
            message: {
              documentMessage: { url: 'https://cdn.example.com/mri_report.pdf', caption: 'Brain MRI Report' }
            }
          }
        }
      }
    }
  }
}, 'documentMessage', (payload) => {
  assert.strictEqual(payload.caption, 'Brain MRI Report');
});

// TC-W11: Quadruple Nested Wrapper (Ephemeral -> Edited -> ViewOnceV2 -> Image)
runTestCase('TC-W11', 'Quadruple Nested Wrapper (Ephemeral -> Edited -> ViewOnceV2 -> Image)', {
  ephemeralMessage: {
    message: {
      editedMessage: {
        message: {
          protocolMessage: {
            editedMessage: {
              viewOnceMessageV2: {
                message: {
                  imageMessage: { url: 'https://cdn.example.com/ct_scan.jpg', caption: 'Abdominal CT Scan' }
                }
              }
            }
          }
        }
      }
    }
  }
}, 'imageMessage', (payload) => {
  assert.strictEqual(payload.caption, 'Abdominal CT Scan');
});

// TC-W12: Null / Undefined Input Handling
runTestCase('TC-W12', 'Null Input Handling', null, null);

console.log('\n--------------------------------------------------');
console.log(`Summary: ${passed} Passed, ${failed} Failed`);
console.log('--------------------------------------------------');

process.exit(failed > 0 ? 1 : 0);
