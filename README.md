# WhatsApp Medical Patient Bot 🩺

Robust WhatsApp Bot for processing medical images, reports, PDFs, audio, and video using Gemini 3.6 Vision AI.

## Key Reliability Features Added:
- **MongoDB Persistent Media Buffering**: Prevents image loss on Render server restarts or idle sleep.
- **Exponential Backoff CDN Retries**: Automatically retries media downloads 5 times on network drops.
- **Startup Recovery**: Resumes processing any un-processed images left in buffer before a restart.
- **Render Health Check Keep-Alive**: Built-in `/health` route compatible with keep-alive uptime pingers.
