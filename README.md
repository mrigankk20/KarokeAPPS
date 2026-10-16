# Smart Hindi Karaoke

Upload a Hindi duet. The app uses AI to **remove the male singer**, keeps the
music and the **female voice**, writes **Hindi (Devanagari) lyrics in sync**, and
lets you **sing the male part** with your own microphone, change the key, add
reverb, and record yourself.

- Real AI source separation (Demucs), not a cheap stereo trick.
- Male/female detection by pitch, with a timeline you can correct by hand.
- Hindi lyrics from the isolated vocals (faster-whisper), written in Devanagari,
  or pulled from LRCLIB / YouTube captions when a match exists.
- Scrolling pitch guide + a singing score, key change, mic effects, recording.

---

## ⚠️ Read this before deploying: Vercel / Netlify will NOT work

This is **not** a static site or a quick serverless function. Removing a singer
is heavy AI work that needs:

- **PyTorch + Demucs** (far over Vercel's 250 MB function limit),
- **minutes per song** (Vercel kills requests in under a minute),
- **a writable disk** (Vercel's filesystem is read-only),
- **ffmpeg**.

So it must run on a host that gives a real container with a disk. This repo is
set up for **Render**, and the same `Dockerfile` also runs on Railway, Fly.io,
Hugging Face Spaces, or any VPS.

---

## Deploy to Render (one click, free-ish)

1. Put these files in a **GitHub repository** (see below).
2. Go to <https://render.com>, sign up, and connect your GitHub.
3. Click **New +  ->  Blueprint**, pick this repo. Render reads `render.yaml`
   and creates the service.
4. Wait for the first build (~10–15 min — it installs PyTorch and builds the
   image). When it says **Live**, open the URL it gives you.
5. The **first song** is slow: Demucs and the lyrics model download their
   weights once (a few hundred MB) onto the attached disk, then stay cached.

**Plan note:** the blueprint uses Render's **Starter** plan (2 GB RAM, a few
dollars a month). Demucs does not fit in the 512 MB **Free** plan — it will be
killed partway. If you only want to try it for free, run it locally (below)
instead, which is also faster on most laptops.

### Putting the files on GitHub

**Easiest (no command line):**
1. Create a new repository at <https://github.com/new>.
2. On the repo page: **Add file  ->  Upload files**.
3. Drag in everything from this folder (`server.py`, `karaoke.html`,
   `Dockerfile`, `render.yaml`, `requirements.txt`, `README.md`, the dotfiles).
4. **Commit changes.** Done — now connect Render as above.

**With git:**
```bash
git init
git add .
git commit -m "Smart Hindi Karaoke"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

---

## Run it on your own computer (free, usually fastest)

```bash
pip install -r requirements.txt      # also install ffmpeg (see below)
python3 server.py
```
Open <http://localhost:8000>.

**ffmpeg** is separate from pip:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: `winget install Gyan.FFmpeg` (or download from ffmpeg.org)

**Sing from your phone** on the same Wi-Fi (the mic needs HTTPS on a phone):
```bash
python3 server.py --host 0.0.0.0 --https
```
Open the `https://<your-computer-ip>:8000` address it prints and accept the
one-time certificate warning.

---

## Other hosts (same repo)

- **Hugging Face Spaces:** create a **Docker** Space, push this repo. Free, and
  a paid GPU tier makes separation take seconds. Set the Space to listen on the
  port it gives you (this app already reads `PORT`).
- **Railway / Fly.io:** point them at the `Dockerfile`; add a volume on
  `/app/cache` so songs and model weights persist.
- **Any VPS (DigitalOcean, Hetzner...):** `docker build -t karaoke . &&
  docker run -p 8000:8000 -v $PWD/cache:/app/cache karaoke`.

---

## Settings (environment variables)

| Variable          | Default            | Meaning |
|-------------------|--------------------|---------|
| `KARAOKE_WHISPER` | `large-v3-turbo` (CPU) / `large-v3` (GPU); repo sets `medium` on Render | Hindi lyrics model. Bigger = better words, more RAM. |
| `KARAOKE_DEMUCS`  | `htdemucs`         | Separation model. `htdemucs_ft` is a little better, slower. |
| `KARAOKE_MAXMIN`  | `15`               | Longest song accepted, in minutes. |
| `PORT`            | set by the host    | When present, the app binds `0.0.0.0` on this port automatically. |

---

## How it works

1. `ffmpeg` decodes the upload to WAV.
2. **Demucs** splits it into `vocals` and `music`.
3. The pitch of the vocals is tracked; each phrase is labelled **male** or
   **female**; the male phrases are muted. Output = music + female voice.
4. **faster-whisper** transcribes the isolated vocals to Devanagari with word
   timing (runs in the background so the song is playable immediately).
5. Key changes, mixing and your recording happen live in the browser.

## Honest limits

- Where both singers overlap on the **same instant**, they cannot be separated;
  that slice is kept or cut as a whole. The in-app timeline lets you correct it.
- AI lyrics are a best effort — Hindi film singing is hard to hear, so expect to
  fix a few words in the edit box when no clean online match exists.

Only use songs you have the right to use.
