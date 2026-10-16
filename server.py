#!/usr/bin/env python3
"""
Smart Hindi Karaoke studio server  (v3)

What it does
  1. Separates a song into vocals and music with Demucs (real AI source separation).
  2. Tracks the pitch of the isolated vocals, works out which phrases are the male singer and which are
     the female singer, and mutes the male phrases.  You hear: music + the original female voice,
     and you sing the male part.  Segments can be corrected by hand in the app.
  3. Writes Hindi (Devanagari) lyrics with word timing from the isolated vocals (faster-whisper).
     This runs AFTER the song is ready, in the background, so you can start singing right away.
  4. Re-renders the song in another key (ffmpeg rubberband, or a fallback if your ffmpeg has none).
  5. Fetches YouTube audio (yt-dlp) and YouTube captions.

Install once:   pip install demucs faster-whisper yt-dlp numpy scipy     (and ffmpeg)
Run:            python3 server.py          then open http://localhost:8000
Phone on Wi-Fi: python3 server.py --host 0.0.0.0 --https     (the microphone needs https on a phone)
Env:            KARAOKE_WHISPER=large-v3|large-v3-turbo|medium   KARAOKE_DEMUCS=htdemucs|htdemucs_ft
                KARAOKE_MAXMIN=15   (longest song accepted, in minutes)
Only use songs you have the right to use.
"""
import argparse
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter, uniform_filter1d

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "cache")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
SR = 44100
HOP = 160                      # 10 ms at 16 kHz
FT = 0.01                      # seconds per pitch frame
PROMPT = "यह हिंदी फिल्मी गाने के बोल हैं।"
MAX_MIN = float(os.environ.get("KARAOKE_MAXMIN", "15"))
JOBS = {}
LOCK = threading.Lock()
CAPS_CACHE = {}
DEVA = re.compile(r"[\u0900-\u097F]")


class JobError(Exception):
    def __init__(self, msg, code="failed", status=500):
        super().__init__(msg)
        self.code, self.status = code, status


def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise JobError("ffmpeg failed: " + p.stderr.decode("utf8", "ignore")[-300:])
    return p


_RB = {}


def has_rubberband():
    if "v" not in _RB:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
            _RB["v"] = b"rubberband" in out
        except Exception:
            _RB["v"] = False
    return _RB["v"]


# ------------------------------------------------------------------ audio io
def read_wav(path):
    sr, a = wavfile.read(path)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    elif a.dtype == np.int32:
        a = a.astype(np.float32) / 2147483648.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 1:
        a = a[:, None]
    return sr, a


def write_wav16(path, sr, a):
    wavfile.write(path, sr, (np.clip(a, -1, 1) * 32767).astype(np.int16))


def to_mp3(wav, mp3):
    tmp = mp3 + ".part.mp3"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "192k", tmp])
    os.replace(tmp, mp3)


def decode(src, dst, sr=SR, ch=2):
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", str(ch), "-ar", str(sr), "-c:a", "pcm_f32le", dst])


def mono16k(src_wav, dst):
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_wav, "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", dst])


# ------------------------------------------------------------------ pitch (YIN)
def track_pitch(x, sr=16000, hop=HOP, fmin=70.0, fmax=900.0, thr=0.18):
    """Frame-wise f0 in Hz (NaN when unvoiced) with the YIN algorithm, vectorised over frames."""
    W = 640                                   # 40 ms integration window
    tmax, tmin = int(sr / fmin) + 1, max(2, int(sr / fmax))
    frame = W + tmax + 2
    x = np.asarray(x, np.float32)
    if len(x) < frame + hop:
        return np.full(1, np.nan, np.float32), np.zeros(1, np.float32)
    n = 1 + (len(x) - frame) // hop
    f0 = np.full(n, np.nan, np.float32)
    rms = np.zeros(n, np.float32)
    nfft = 2048
    taus = np.arange(1, tmax + 1, dtype=np.float32)
    for b0 in range(0, n, 1200):
        idx = np.arange(b0, min(n, b0 + 1200))
        fr = np.stack([x[i * hop:i * hop + frame] for i in idx])
        head = fr[:, :W]
        e_head = np.sum(head * head, axis=1)
        rms[idx] = np.sqrt(e_head / W)
        acf = np.fft.irfft(np.conj(np.fft.rfft(head, nfft)) * np.fft.rfft(fr, nfft), nfft)[:, :tmax + 1]
        sq = np.concatenate([np.zeros((len(idx), 1), np.float32), np.cumsum(fr * fr, axis=1)], axis=1)
        e_shift = sq[:, W:W + tmax + 1] - sq[:, :tmax + 1]
        d = e_head[:, None] + e_shift - 2 * acf
        d[:, 0] = 0
        cm = np.ones_like(d)
        cm[:, 1:] = d[:, 1:] * taus[None, :] / (np.cumsum(d[:, 1:], axis=1) + 1e-9)
        sub = cm[:, tmin:tmax]
        below = sub < thr
        has = below.any(axis=1)
        first = np.argmax(below, axis=1)
        win = 30
        gi = np.clip(first[:, None] + np.arange(win)[None, :], 0, sub.shape[1] - 1)
        vals = np.take_along_axis(sub, gi, axis=1)
        up = np.diff(vals, axis=1) >= 0
        stop = np.where(up.any(axis=1), np.argmax(up, axis=1), win - 1)
        rows = np.arange(len(idx))
        tau_a = tmin + np.clip(first + stop, 0, sub.shape[1] - 1)
        gmin_i = np.argmin(sub, axis=1)
        gmin_v = sub[rows, gmin_i]
        tau = np.where(has, tau_a, tmin + gmin_i)
        voiced = has | (gmin_v < 0.25)
        t = np.clip(tau, 1, tmax - 1)
        a, bb, c = cm[rows, t - 1], cm[rows, t], cm[rows, t + 1]
        den = a - 2 * bb + c
        sh = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(np.abs(den) > 1e-9, den, 1), 0)
        tau_f = t + np.clip(sh, -1, 1)
        f0[idx] = np.where(voiced, sr / tau_f, np.nan)
    gate = 0.04 * (np.percentile(rms, 95) + 1e-9)
    f0[rms < gate] = np.nan
    return f0, rms


def f0_to_midi(f0):
    """MIDI note per frame, NaN when unvoiced; removes isolated octave errors and spikes."""
    m = np.full(len(f0), np.nan, np.float32)
    ok = ~np.isnan(f0)
    m[ok] = 69 + 12 * np.log2(f0[ok] / 440.0)
    if ok.sum() < 20:
        return m
    idx = np.arange(len(m))
    filled = np.interp(idx, idx[ok], m[ok])                      # nearest-ish fill for the filter only
    med = median_filter(filled, size=11, mode="nearest")
    bad = ok & (np.abs(m - med) > 3.0)
    m[bad] = np.nan
    return m


# ------------------------------------------------------------------ male / female labelling
def _runs(labels):
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((s, i, labels[s]))
            s = i
    return out


def voiced_pieces(midi, bridge=8, minlen=8, maxlen=100):
    """Cut the voiced frames into pieces of at most 1 s: [(a, b, median_midi, seconds, gap_before_s)]."""
    v = ~np.isnan(midi)
    runs = [(a, b) for a, b, l in _runs(v.astype(int).tolist()) if l == 1]
    merged = []
    for a, b in runs:                              # bridge short unvoiced holes (consonants, breaths)
        if merged and a - merged[-1][1] <= bridge:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    pieces, prev_end = [], 0
    for a, b in merged:
        if int(v[a:b].sum()) < minlen:
            continue
        k = max(1, int(np.ceil((b - a) / maxlen)))
        edges = np.linspace(a, b, k + 1).astype(int)
        for j in range(k):
            lo, hi = int(edges[j]), int(edges[j + 1])
            vals = midi[lo:hi][v[lo:hi]]
            if len(vals) < 4:
                continue
            gap = (lo - prev_end) * FT if j == 0 else 0.0
            pieces.append((lo, hi, float(np.median(vals)), len(vals) * FT, gap))
            prev_end = hi
    return pieces


def _kmeans2(m, w):
    c1, c2 = np.percentile(m, 20), np.percentile(m, 80)
    if c2 - c1 < 0.5:
        c1, c2 = m.min(), m.max()
    for _ in range(30):
        hi = np.abs(m - c2) < np.abs(m - c1)
        if hi.all() or (~hi).all():
            break
        n1, n2 = np.average(m[~hi], weights=w[~hi]), np.average(m[hi], weights=w[hi])
        if abs(n1 - c1) + abs(n2 - c2) < 1e-3:
            c1, c2 = n1, n2
            break
        c1, c2 = n1, n2
    return float(c1), float(c2)


def _bimodal(m, w, c1, c2):
    """True when the pitch histogram really has two humps (a duet) rather than one wide hump."""
    bins = np.arange(m.min() - 4, m.max() + 5, 0.5)
    h, e = np.histogram(m, bins=bins, weights=w)
    k = np.exp(-0.5 * (np.arange(-8, 9) * 0.5 / 1.2) ** 2)
    h = np.convolve(h, k / k.sum(), mode="same")
    ctr = (e[:-1] + e[1:]) / 2
    lo = (ctr >= c1 - 3) & (ctr <= c1 + 3)
    hi = (ctr >= c2 - 3) & (ctr <= c2 + 3)
    if not lo.any() or not hi.any():
        return False
    p1, p2 = h[lo].max(), h[hi].max()
    between = (ctr > c1) & (ctr < c2)
    valley = h[between].min() if between.any() else min(p1, p2)
    return valley < 0.62 * min(p1, p2)


def label_voices(midi, mode="auto", split_hz=None):
    """Return (labels per 10 ms frame: 1 = female / high voice, 0 = male / low voice, info)."""
    n = len(midi)
    info = {"duet": False, "male_hz": None, "female_hz": None, "split_hz": None, "female_ok": True, "mode": mode}
    pcs = voiced_pieces(midi)
    if len(pcs) < 4:
        info["note"] = "no pitch found"
        return np.ones(n, np.uint8), info
    m = np.array([p[2] for p in pcs])
    w = np.array([p[3] for p in pcs])
    hz = lambda x: int(round(440.0 * 2 ** ((x - 69) / 12.0)))
    duet = False
    c1 = c2 = None
    if split_hz:
        thr = 69 + 12 * np.log2(split_hz / 440.0)
        lo, hi = m <= thr, m > thr
        if lo.any() and hi.any():
            c1, c2 = float(np.average(m[lo], weights=w[lo])), float(np.average(m[hi], weights=w[hi]))
            duet = True
        info["mode"] = "duet"
    elif mode in ("auto", "duet"):
        c1, c2 = _kmeans2(m, w)
        share = w[np.abs(m - c2) < np.abs(m - c1)].sum() / w.sum()
        if mode == "duet":
            duet = c2 - c1 >= 2.0
        else:
            duet = (c2 - c1 >= 5.0 and 0.12 <= share <= 0.88 and c1 < 58 and _bimodal(m, w, c1, c2))
    if not duet:
        overall = float(np.average(m, weights=w))
        male = (mode == "male") or (mode == "auto" and overall < 54.0) or (mode == "duet" and overall < 54.0)
        if mode == "female":
            male = False
        info["duet"] = False
        info["split_hz"] = None
        if male:
            info["female_ok"] = False
            info["male_hz"] = hz(overall)
            return np.zeros(n, np.uint8), info
        info["female_hz"] = hz(overall)
        return np.ones(n, np.uint8), info
    # two singers: Viterbi over pieces, with switch costs that depend on the pitch jump and the pause
    sel = (np.abs(m - c1) < np.abs(m - c2))
    s1 = float(np.sqrt(np.average((m[sel] - c1) ** 2, weights=w[sel]))) if sel.any() else 3.0
    s2 = float(np.sqrt(np.average((m[~sel] - c2) ** 2, weights=w[~sel]))) if (~sel).any() else 3.0
    sig = float(np.clip((s1 + s2) / 2, 2.0, 3.0))
    cost = np.stack([0.5 * ((m - c1) / sig) ** 2, 0.5 * ((m - c2) / sig) ** 2], axis=1) * np.clip(w, 0.35, 1.0)[:, None]
    N = len(pcs)
    best = np.zeros((N, 2))
    back = np.zeros((N, 2), np.int8)
    best[0] = cost[0]
    for i in range(1, N):
        gap = pcs[i][4]
        base = 0.5 if gap >= 1.0 else (0.9 if gap >= 0.5 else (1.6 if gap >= 0.25 else 3.2))
        jump = m[i] - m[i - 1]
        for s in (0, 1):
            if s == 0:      # staying male / coming back to male
                stay = best[i - 1][0] + 0.35 * max(0.0, abs(jump) - 5.0)
                sw = best[i - 1][1] + base * (max(0.15, 1 + jump / 9.0) if jump < 0 else 1.0)
            else:           # staying female / going to female
                stay = best[i - 1][1] + 0.35 * max(0.0, abs(jump) - 5.0)
                sw = best[i - 1][0] + base * (max(0.15, 1 - jump / 9.0) if jump > 0 else 1.0)
            if stay <= sw:
                best[i][s], back[i][s] = stay + cost[i][s], s
            else:
                best[i][s], back[i][s] = sw + cost[i][s], 1 - s
    st = np.zeros(N, np.int8)
    st[-1] = int(np.argmin(best[-1]))
    for i in range(N - 1, 0, -1):
        st[i - 1] = back[i][st[i]]
    lab = np.zeros(n, np.uint8)
    lab[:pcs[0][0]] = st[0]
    for i, (a, b, _, _, _) in enumerate(pcs):
        lab[a:b] = st[i]
        nxt = pcs[i + 1][0] if i + 1 < N else n
        mid = (b + nxt) // 2 if i + 1 < N else n
        lab[b:mid] = st[i]
        if i + 1 < N:
            lab[mid:nxt] = st[i + 1]
    info.update(duet=True, male_hz=hz(c1), female_hz=hz(c2), split_hz=hz((c1 + c2) / 2.0), mode="duet")
    return lab, info


def labels_to_segments(lab, total_s, min_len=0.25):
    segs = [[round(a * FT, 2), round(b * FT, 2), "f" if l else "m"] for a, b, l in _runs(lab.astype(int).tolist())]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s[1] - s[0] < min_len:
                j = i - 1 if i > 0 else i + 1
                segs[j][0], segs[j][1] = min(segs[j][0], s[0]), max(segs[j][1], s[1])
                segs.pop(i)
                changed = True
                break
        k = 0
        while k + 1 < len(segs):
            if segs[k][2] == segs[k + 1][2]:
                segs[k][1] = segs[k + 1][1]
                segs.pop(k + 1)
                changed = True
            else:
                k += 1
    if segs:
        segs[0][0] = 0.0
        segs[-1][1] = round(total_s, 2)
    return segs


def segments_to_mask(segs, nsamples, sr=SR, fade=0.06):
    m = np.zeros(nsamples, np.float32)
    for s, e, l in segs:
        if l == "f":
            m[max(0, int(s * sr)):min(nsamples, int(e * sr))] = 1.0
    return uniform_filter1d(m, max(1, int(fade * sr)))


def notes_from_midi(midi, lab):
    """Group frames into note bars for the pitch lane: [start_s, end_s, midi, 'm'|'f']."""
    out, n, i = [], len(midi), 0
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        j, acc = i, [midi[i]]
        while j + 1 < n and not np.isnan(midi[j + 1]) and abs(midi[j + 1] - np.median(acc)) < 0.9 and lab[j + 1] == lab[i]:
            j += 1
            acc.append(midi[j])
        if len(acc) >= 8:
            out.append([round(i * FT + 0.03, 2), round((j + 1) * FT + 0.03, 2), round(float(np.median(acc)), 1), "f" if lab[i] else "m"])
        i = j + 1
    return out


# ------------------------------------------------------------------ AI steps
def device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def separate(job, wav_path, d):
    """Demucs in 35 s chunks with cross-fades so progress can be shown."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    job_stage(job, 8, "Loading the separation model (first run downloads about 80 MB)")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = get_model(os.environ.get("KARAOKE_DEMUCS", "htdemucs"))
    model.eval()
    dev = device()
    sr, a = read_wav(wav_path)
    n = a.shape[0]
    wav = torch.from_numpy(np.ascontiguousarray(a.T))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    chunk, ov = int(35 * sr), int(1.5 * sr)
    starts, s = [], 0
    while True:
        starts.append(s)
        if s + chunk >= n:
            break
        s += chunk - ov
    voc = np.zeros((n, 2), np.float32)
    mus = np.zeros((n, 2), np.float32)
    ws = np.zeros(n, np.float32)
    vi = model.sources.index("vocals")
    t0 = time.time()
    for k, st in enumerate(starts):
        check_cancel(job)
        piece = wav[:, st:st + chunk]
        L = piece.shape[1]
        try:
            src = apply_model(model, piece[None], device=dev, shifts=0, split=True, overlap=0.1, progress=False)[0]
        except RuntimeError:
            if dev == "cpu":
                raise
            dev = "cpu"
            src = apply_model(model, piece[None], device=dev, shifts=0, split=True, overlap=0.1, progress=False)[0]
        src = (src * std + mean).cpu().numpy()
        v = src[vi].T
        m = src.sum(0).T - v
        w = np.ones(L, np.float32)
        if k > 0:
            r = min(ov, L)
            w[:r] = np.linspace(0, 1, r, dtype=np.float32)
        if k < len(starts) - 1:
            r = min(ov, L)
            w[L - r:] = np.minimum(w[L - r:], np.linspace(1, 0, r, dtype=np.float32))
        voc[st:st + L] += v * w[:, None]
        mus[st:st + L] += m * w[:, None]
        ws[st:st + L] += w
        done = k + 1
        el = time.time() - t0
        job["eta"] = int(el / done * (len(starts) - done))
        job_stage(job, 10 + int(48 * done / len(starts)), "Separating voice from music (%d of %d)" % (done, len(starts)))
    ws = np.maximum(ws, 1e-6)[:, None]
    voc /= ws
    mus /= ws
    write_wav16(os.path.join(d, "vocals.wav"), sr, voc)
    write_wav16(os.path.join(d, "music.wav"), sr, mus)


def whisper_name(dev):
    return os.environ.get("KARAOKE_WHISPER") or ("large-v3" if dev == "cuda" else "large-v3-turbo")


def load_whisper(job, dev):
    from faster_whisper import WhisperModel

    size = whisper_name(dev)
    ctype = "float16" if dev == "cuda" else "int8"
    job_stage(job, 6, "Loading the Hindi lyrics model '%s' (first run downloads it)" % size)
    tried = [size] + [x for x in (("large-v3",) if dev == "cuda" else ("medium",)) if x != size]
    last = None
    for name in tried:
        try:
            return WhisperModel(name, device=dev, compute_type=ctype, cpu_threads=os.cpu_count() or 4), name
        except Exception as e:
            last = e
            print("whisper model %s failed: %s" % (name, e), file=sys.stderr)
    raise JobError("Could not load the lyrics model: %s" % last, "nolyrics")


def latin_token_ids(model):
    """Token ids that spell plain English letters. Banning them keeps Whisper writing Devanagari."""
    try:
        vocab = model.hf_tokenizer.get_vocab()
    except Exception:
        return []
    ids = []
    for tok, i in vocab.items():
        if i >= 50257 or tok.startswith("<|"):
            continue
        s = tok.replace("\u0120", "")
        if s and s.isascii() and re.search(r"[A-Za-z]", s):
            ids.append(int(i))
    return ids


def transcribe(job, wav16, duration, dev):
    model, name = load_whisper(job, dev)
    ban = latin_token_ids(model)
    job_stage(job, 10, "Writing Hindi lyrics from the vocals")
    kw = dict(language="hi", task="transcribe", beam_size=5 if dev == "cuda" else 3, word_timestamps=True, vad_filter=True,
              vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 250}, condition_on_previous_text=False,
              initial_prompt=PROMPT, temperature=[0.0, 0.2, 0.4], compression_ratio_threshold=2.0, no_speech_threshold=0.5,
              log_prob_threshold=-1.2)
    try:
        segs, _ = model.transcribe(wav16, suppress_tokens=[-1] + ban, **kw)
        segs = list(_progress(job, segs, duration))
    except Exception as e:
        if not ban:
            raise
        print("lyrics retry without token ban:", e, file=sys.stderr)
        segs, _ = model.transcribe(wav16, **kw)
        segs = list(_progress(job, segs, duration))
    return build_lines(segs_to_words(segs))


def _progress(job, segs, duration):
    for sg in segs:
        check_cancel(job)
        job_stage(job, 10 + int(88 * min(1.0, sg.end / max(duration, 1))), "Writing Hindi lyrics from the vocals")
        yield sg


def segs_to_words(segs):
    """One list of words per Whisper segment; segments that are not Hindi or are hallucinations are dropped."""
    out = []
    for sg in segs:
        if sg.no_speech_prob > 0.8 or sg.avg_logprob < -1.6:
            continue
        txt = sg.text or ""
        if not DEVA.search(txt):
            continue
        if getattr(sg, "compression_ratio", 0) and sg.compression_ratio > 2.6:
            continue
        ws = []
        for w in sg.words or []:
            t = re.sub(r"[\s\"“”.,;:!?|]+", "", w.word)
            if t and DEVA.search(t):
                ws.append({"w": t, "s": round(w.start, 2), "e": round(max(w.end, w.start + 0.12), 2)})
        if ws:
            out.append(ws)
    return out


def build_lines(groups):
    lines = []
    for ws in groups:
        cur = [ws]
        while True:
            long_ = [g for g in cur if len(g) > 10 or g[-1]["e"] - g[0]["s"] > 8]
            if not long_:
                break
            g = long_[0]
            k = max(range(1, len(g)), key=lambda i: g[i]["s"] - g[i - 1]["e"] + (0.3 if 3 <= i <= len(g) - 3 else 0))
            i = cur.index(g)
            cur[i:i + 1] = [g[:k], g[k:]]
        lines.extend(cur)
    out = []
    for ln in lines:
        text = " ".join(x["w"] for x in ln)
        if len(out) >= 2 and out[-1]["text"] == text == out[-2]["text"]:
            continue
        out.append({"t": ln[0]["s"], "e": ln[-1]["e"], "text": text, "words": ln})
    for i in range(len(out) - 1):
        if out[i]["e"] > out[i + 1]["t"]:
            out[i]["e"] = max(out[i]["t"] + 0.3, out[i + 1]["t"] - 0.02)
    return out


def lrclib(title, artist, duration):
    qs = []
    for q in ((title + " " + artist).strip(), title.strip()):
        if q and q not in qs:
            qs.append(q)
    for q in qs:
        try:
            raw = urllib.request.urlopen(urllib.request.Request("https://lrclib.net/api/search?q=" + urllib.parse.quote(q), headers={"User-Agent": UA}), timeout=8).read()
            items = json.loads(raw)
        except Exception:
            continue
        for x in items:
            txt = x.get("syncedLyrics")
            if not txt or len(DEVA.findall(txt)) < 40:
                continue
            if x.get("duration") and duration and abs(x["duration"] - duration) > 4.0:
                continue
            return txt
    return None


# ------------------------------------------------------------------ job helpers
def job_stage(job, pct, stage):
    job["pct"], job["stage"] = pct, stage


def check_cancel(job):
    if job.get("cancel"):
        raise JobError("Cancelled", "cancelled")


def start_job(fn, *args):
    now = time.time()
    for k in [k for k, j in JOBS.items() if j.get("state") in ("done", "error") and now - j.get("t1", now) > 3600]:
        JOBS.pop(k, None)
    jid = uuid.uuid4().hex[:10]
    job = {"state": "queued", "stage": "Waiting for the server", "pct": 0, "t0": now}
    JOBS[jid] = job

    def runner():
        with LOCK:
            job["state"], job["t0"] = "running", time.time()
            try:
                job["result"] = fn(job, *args)
                job.update(state="done", pct=100, stage="Done", t1=time.time())
            except JobError as e:
                job.update(state="error", error=str(e), code=e.code, t1=time.time())
            except Exception as e:
                traceback.print_exc()
                job.update(state="error", error="%s: %s" % (type(e).__name__, e), code="failed", t1=time.time())

    threading.Thread(target=runner, daemon=True).start()
    return jid


def cdir(cid):
    d = os.path.join(CACHE_DIR, cid)
    os.makedirs(d, exist_ok=True)
    return d


def load_meta(d):
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_meta(d, m):
    with open(os.path.join(d, "meta.json"), "w", encoding="utf8") as f:
        json.dump(m, f, ensure_ascii=False)


def jload(d, name, default=None):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return default
    try:
        with open(p, encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return default


def jsave(d, name, obj):
    with open(os.path.join(d, name), "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False)


def url_of(cid, name, d):
    p = os.path.join(d, name)
    return "/cache/%s/%s?v=%d" % (cid, name, int(os.path.getmtime(p))) if os.path.exists(p) else None


def build_result(cid):
    d = cdir(cid)
    m = load_meta(d)
    online = None
    if os.path.exists(os.path.join(d, "online.lrc")):
        with open(os.path.join(d, "online.lrc"), encoding="utf8") as f:
            online = f.read()
    keys = {}
    for f in os.listdir(d):
        r = re.match(r"(karaoke|music|original)_s(-?\d+)\.mp3$", f)
        if r:
            keys.setdefault(int(r.group(2)), {})[r.group(1)] = url_of(cid, f, d)
    return {
        "id": cid, "title": m.get("title", ""), "artist": m.get("artist", ""), "duration": m.get("duration", 0),
        "duet": m.get("duet", False), "female_ok": m.get("female_ok", True), "mode": m.get("mode", "auto"),
        "male_hz": m.get("male_hz"), "female_hz": m.get("female_hz"), "split_hz": m.get("split_hz"),
        "edited": m.get("edited", False), "guide": m.get("guide", 0),
        "urls": {k: url_of(cid, k + ".mp3", d) for k in ("karaoke", "music", "original")},
        "shifted": keys,
        "segments": jload(d, "segments.json", []), "notes": jload(d, "notes.json", []),
        "lyrics_ai": jload(d, "lyrics_ai.json", None), "lyrics_state": m.get("lyrics_state", ""),
        "online_lrc": online, "rubberband": has_rubberband(),
    }


def clean_title(n):
    n = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", n)
    n = re.sub(r"[\(\[][^\)\]]*(official|video|lyric|audio|hd|4k|song|full)[^\)\]]*[\)\]]", " ", n, flags=re.I)
    n = re.sub(r"snapyt[ _.-]*app|snapyt|y2mate(\.com)?|savefrom(\.net)?|ssyoutube|ytmp3|ytdown|\(\d+\)|\bcopy\b", " ", n, flags=re.I)
    n = re.sub(r"\b(official|full video|lyrical video|lyric video|video song|audio song|hd|4k|1080p|720p)\b", " ", n, flags=re.I)
    n = re.sub(r"[_.]+", " ", n)
    if " " not in n.strip():
        n = n.replace("-", " ")
    else:
        n = re.sub(r"\s*-\s*", " ", n)
    return re.sub(r"\s+", " ", n).strip(" -|")


# ------------------------------------------------------------------ pipeline
def analyse_voices(job, cid, mode="auto", hz=None):
    d = cdir(cid)
    job_stage(job, 60, "Finding the male and female voices")
    f0_path = os.path.join(d, "f0.npy")
    if os.path.exists(f0_path):
        f0 = np.load(f0_path)
    else:
        v16 = os.path.join(d, "v16.wav")
        mono16k(os.path.join(d, "vocals.wav"), v16)
        _, x = read_wav(v16)
        f0, _ = track_pitch(x[:, 0])
        np.save(f0_path, f0)
    midi = f0_to_midi(f0)
    lab, info = label_voices(midi, mode, hz)
    meta = load_meta(d)
    total = meta.get("duration", len(f0) * FT)
    segs = labels_to_segments(lab, total)
    jsave(d, "segments.json", segs)
    jsave(d, "notes.json", notes_from_midi(midi, lab))
    meta.update(duet=info["duet"], female_ok=info["female_ok"], male_hz=info.get("male_hz"), female_hz=info.get("female_hz"),
                split_hz=info.get("split_hz"), mode=info.get("mode", mode), edited=False)
    save_meta(d, meta)
    return segs, info


def mix_karaoke(job, cid, segs, guide=0.0):
    """music + vocals, where the male phrases are muted (or kept at a low guide level)."""
    d = cdir(cid)
    job_stage(job, 72, "Removing the male voice")
    sr, voc = read_wav(os.path.join(d, "vocals.wav"))
    _, mus = read_wav(os.path.join(d, "music.wav"))
    n = min(len(voc), len(mus))
    voc, mus = voc[:n], mus[:n]
    m = segments_to_mask(segs, n, sr)
    g = float(np.clip(guide, 0, 1))
    gain = g + (1 - g) * m
    mix = mus + voc * gain[:, None]
    pk = float(np.max(np.abs(mix))) or 1.0
    if pk > 0.98:
        mix *= 0.98 / pk
    write_wav16(os.path.join(d, "karaoke.wav"), sr, mix)
    for f in os.listdir(d):
        if re.match(r"karaoke_s-?\d+\.mp3$", f):
            os.remove(os.path.join(d, f))
    job_stage(job, 82, "Saving the karaoke track")
    to_mp3(os.path.join(d, "karaoke.wav"), os.path.join(d, "karaoke.mp3"))


def process(job, cid, lyrics=True, title="", artist=""):
    d = cdir(cid)
    meta = load_meta(d)
    if meta.get("done") and os.path.exists(os.path.join(d, "karaoke.mp3")):
        job_stage(job, 99, "Found this song in the cache")
        return build_result(cid)
    src = next((os.path.join(d, f) for f in os.listdir(d) if f.startswith("src.")), None)
    if not src:
        raise JobError("The source file is missing.")
    if title:
        meta["title"] = title
    if artist:
        meta["artist"] = artist
    job_stage(job, 3, "Reading the audio")
    mix = os.path.join(d, "mix.wav")
    if not os.path.exists(mix):
        decode(src, mix)
    sr, a = read_wav(mix)
    meta["duration"] = round(a.shape[0] / sr, 2)
    del a
    if meta["duration"] > MAX_MIN * 60:
        raise JobError("This song is %.0f minutes long. The limit is %.0f minutes (set KARAOKE_MAXMIN to change it)." % (meta["duration"] / 60, MAX_MIN), "toolong", 400)
    save_meta(d, meta)
    if not os.path.exists(os.path.join(d, "original.mp3")):
        to_mp3(mix, os.path.join(d, "original.mp3"))
    if not (os.path.exists(os.path.join(d, "vocals.wav")) and os.path.exists(os.path.join(d, "music.wav"))):
        separate(job, mix, d)
    check_cancel(job)
    segs, _ = analyse_voices(job, cid)
    mix_karaoke(job, cid, segs, 0.0)
    job_stage(job, 90, "Saving the music-only track")
    to_mp3(os.path.join(d, "music.wav"), os.path.join(d, "music.mp3"))
    try:
        os.remove(mix)
    except OSError:
        pass
    meta = load_meta(d)
    meta["done"] = True
    save_meta(d, meta)
    job_stage(job, 95, "Looking for matching online lyrics")
    if not os.path.exists(os.path.join(d, "online.lrc")):
        lrc = lrclib(meta.get("title", ""), meta.get("artist", ""), meta.get("duration", 0))
        if lrc:
            with open(os.path.join(d, "online.lrc"), "w", encoding="utf8") as f:
                f.write(lrc)
    return build_result(cid)


def job_lyrics(job, cid):
    d = cdir(cid)
    if not have("faster_whisper"):
        raise JobError("Install the lyrics model first: pip install faster-whisper", "nolyrics")
    meta = load_meta(d)
    meta["lyrics_state"] = "running"
    save_meta(d, meta)
    try:
        v16 = os.path.join(d, "v16.wav")
        if not os.path.exists(v16):
            mono16k(os.path.join(d, "vocals.wav"), v16)
        lines = transcribe(job, v16, meta.get("duration", 0), device())
        jsave(d, "lyrics_ai.json", lines)
        meta = load_meta(d)
        meta["lyrics_state"] = "done" if lines else "empty"
    except BaseException:
        meta = load_meta(d)
        meta["lyrics_state"] = "failed"
        raise
    finally:
        save_meta(d, meta)
    return build_result(cid)


def job_split(job, cid, mode, hz):
    segs, _ = analyse_voices(job, cid, mode, hz)
    mix_karaoke(job, cid, segs, load_meta(cdir(cid)).get("guide", 0))
    return build_result(cid)


def job_remix(job, cid, segs, guide):
    d = cdir(cid)
    clean = []
    for s in segs:
        try:
            a, b, l = float(s[0]), float(s[1]), ("f" if s[2] == "f" else "m")
        except Exception:
            continue
        if b > a:
            clean.append([round(a, 2), round(b, 2), l])
    if not clean:
        raise JobError("No segments given.")
    jsave(d, "segments.json", clean)
    meta = load_meta(d)
    meta.update(edited=True, guide=round(float(guide), 2))
    save_meta(d, meta)
    mix_karaoke(job, cid, clean, guide)
    return build_result(cid)


def shift_cmd(src, dst, st):
    ratio = 2 ** (st / 12.0)
    if has_rubberband():
        af = "rubberband=pitch=%.6f:formant=preserved:pitchq=quality" % ratio
    else:                                   # works in every ffmpeg, a little lower quality
        af = "asetrate=%d,aresample=%d,atempo=%.6f" % (round(SR * ratio), SR, 1 / ratio)
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, "-codec:a", "libmp3lame", "-b:a", "192k", dst]


def job_render(job, cid, st, which):
    d = cdir(cid)
    if which not in ("karaoke", "music", "original"):
        raise JobError("Unknown track")
    src = os.path.join(d, "music.wav") if which == "music" else os.path.join(d, which + ".wav")
    if not os.path.exists(src):
        src = os.path.join(d, which + ".mp3")
    dst = os.path.join(d, "%s_s%d.mp3" % (which, st))
    job_stage(job, 15, "Changing the key (%+d semitones)" % st)
    if not os.path.exists(dst):
        tmp = dst + ".part.mp3"
        run(shift_cmd(src, tmp, st))
        os.replace(tmp, dst)
    return {"key": st, "which": which, "url": url_of(cid, os.path.basename(dst), d)}


def job_file(job, cid, lyrics, title, artist):
    return process(job, cid, lyrics, title, artist)


def job_url(job, url, lyrics):
    import yt_dlp

    job_stage(job, 2, "Downloading the song from YouTube")
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([\w-]{11})", url)
    vid = m.group(1) if m else hashlib.sha1(url.encode()).hexdigest()[:11]
    cid = "yt-" + vid
    d = cdir(cid)
    meta = load_meta(d)
    if not any(f.startswith("src.") for f in os.listdir(d)):
        opts = {"format": "bestaudio/best", "outtmpl": os.path.join(d, "src.%(ext)s"), "noplaylist": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=True)
        meta["title"] = clean_title(info.get("track") or info.get("title") or "")
        meta["artist"] = re.sub(r" - Topic$", "", info.get("artist") or info.get("uploader") or "")
        meta["vid"] = vid
        save_meta(d, meta)
    return process(job, cid, lyrics, meta.get("title", ""), meta.get("artist", ""))


def library():
    out = []
    if not os.path.isdir(CACHE_DIR):
        return out
    for cid in os.listdir(CACHE_DIR):
        d = os.path.join(CACHE_DIR, cid)
        m = load_meta(d) if os.path.isdir(d) else {}
        if m.get("done"):
            out.append({"id": cid, "title": m.get("title", ""), "artist": m.get("artist", ""), "duration": m.get("duration", 0),
                        "duet": m.get("duet", False), "t": os.path.getmtime(os.path.join(d, "meta.json"))})
    return sorted(out, key=lambda x: -x["t"])[:30]


# ------------------------------------------------------------------ YouTube captions
def http(url, data=None, headers=None, timeout=25):
    req = urllib.request.Request(url, data=data, headers=headers or {"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class CapError(Exception):
    def __init__(self, code, msg, status=502):
        super().__init__(msg)
        self.code, self.msg, self.status = code, msg, status


def tracks_ytdlp(vid):
    import yt_dlp

    with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True, "noplaylist": True}) as y:
        info = y.extract_info("https://www.youtube.com/watch?v=" + vid, download=False)
    out = []
    for kind, src in (("manual", info.get("subtitles") or {}), ("auto", info.get("automatic_captions") or {})):
        for lang, fmts in src.items():
            f = next((x for x in fmts if x.get("ext") == "json3"), None)
            if lang == "live_chat" or not f or (kind == "auto" and "tlang=" in f["url"]):
                continue
            out.append({"lang": lang, "kind": kind, "url": f["url"], "name": f.get("name") or lang})
    return info.get("title"), out


def tracks_innertube(vid):
    payload = {"context": {"client": {"clientName": "ANDROID", "clientVersion": "20.10.38", "androidSdkVersion": 30, "hl": "en", "gl": "IN"}}, "videoId": vid}
    headers = {"Content-Type": "application/json", "User-Agent": "com.google.android.youtube/20.10.38 (Linux; U; Android 11) gzip"}
    j = json.loads(http("https://www.youtube.com/youtubei/v1/player?prettyPrint=false", json.dumps(payload).encode(), headers))
    caps = (((j.get("captions") or {}).get("playerCaptionsTracklistRenderer")) or {}).get("captionTracks") or []
    out = []
    for c in caps:
        nm = c.get("name") or {}
        label = nm.get("simpleText") or "".join(r.get("text", "") for r in nm.get("runs", [])) or c.get("languageCode")
        out.append({"lang": c.get("languageCode", "und"), "kind": "auto" if c.get("kind") == "asr" else "manual", "url": c["baseUrl"], "name": label})
    return (j.get("videoDetails") or {}).get("title"), out


def list_tracks(vid):
    try:
        return tracks_ytdlp(vid)
    except ImportError:
        pass
    except Exception as e:
        print("yt-dlp failed:", e, file=sys.stderr)
    try:
        return tracks_innertube(vid)
    except Exception as e:
        raise CapError("blocked", "Could not reach YouTube: %s" % e)


def choose(tracks, want=None):
    key = lambda t: "%s:%s" % (t["lang"], t["kind"])
    for t in tracks:
        if want and key(t) == want:
            return t
    hindi = [t for t in tracks if t["lang"].lower().startswith("hi")]
    for pool in (hindi, tracks):
        for kind in ("manual", "auto"):
            for t in pool:
                if t["kind"] == kind:
                    return t
    return None


def get_captions(vid, want=None):
    if not re.fullmatch(r"[\w-]{11}", vid):
        raise CapError("badid", "That is not a YouTube video id.", 400)
    ck = (vid, want or "")
    if ck in CAPS_CACHE:
        return CAPS_CACHE[ck]
    title, tracks = list_tracks(vid)
    if not tracks:
        raise CapError("nocaps", "No captions on this video.", 404)
    t = choose(tracks, want)
    url = t["url"]
    url = re.sub(r"([?&])fmt=[^&]*", r"\1fmt=json3", url) if "fmt=" in url else url + "&fmt=json3"
    body = http(url)
    if not body.strip():
        raise CapError("blocked", "YouTube returned empty captions.")
    try:
        j3 = json.loads(body)
    except ValueError:
        raise CapError("blocked", "YouTube returned captions in an unexpected format.")
    res = {"title": title, "chosen": "%s:%s" % (t["lang"], t["kind"]),
           "tracks": [{"key": "%s:%s" % (x["lang"], x["kind"]), "label": "%s%s" % (x["name"], " (auto-generated)" if x["kind"] == "auto" else "")} for x in tracks],
           "json3": j3}
    CAPS_CACHE[ck] = res
    return res


# ------------------------------------------------------------------ http
def valid_id(cid):
    return bool(re.fullmatch(r"[\w-]{4,40}", cid or "")) and os.path.isdir(os.path.join(CACHE_DIR, cid))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path):
        if not os.path.isfile(path):
            return self.send_json({"error": "Not found"}, 404)
        size = os.path.getsize(path)
        start, end, status = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1) == "":
                start = max(0, size - int(m.group(2)))
            else:
                start = int(m.group(1))
                if m.group(2):
                    end = min(size - 1, int(m.group(2)))
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if status == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    buf = f.read(min(65536, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
            pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        p = u.path
        try:
            if p == "/api/ping":
                feats = {"demucs": have("demucs"), "whisper": have("faster_whisper"), "ytdlp": have("yt_dlp"),
                         "ffmpeg": bool(shutil.which("ffmpeg")), "rubberband": has_rubberband(), "device": device() if have("torch") else "cpu",
                         "whisper_model": whisper_name(device() if have("torch") else "cpu")}
                return self.send_json({"ok": True, "version": 3, "features": feats})
            if p == "/api/job":
                j = JOBS.get(q.get("id", ""))
                if not j:
                    return self.send_json({"error": "Unknown job"}, 404)
                out = {k: v for k, v in j.items() if k not in ("t0", "t1")}
                out["elapsed"] = round((j.get("t1") or time.time()) - j["t0"])
                return self.send_json(out)
            if p == "/api/cancel":
                if q.get("id") in JOBS:
                    JOBS[q["id"]]["cancel"] = True
                return self.send_json({"ok": True})
            if p == "/api/library":
                return self.send_json({"songs": library()})
            if p == "/api/captions":
                try:
                    return self.send_json(get_captions(q.get("v", ""), q.get("track", "")))
                except CapError as e:
                    return self.send_json({"error": e.msg, "code": e.code}, e.status)
            if p in ("/api/song", "/api/split", "/api/lyrics", "/api/render", "/api/delete"):
                cid = q.get("id", "")
                if not valid_id(cid):
                    return self.send_json({"error": "Unknown song", "code": "nosong"}, 404)
                if p == "/api/song":
                    return self.send_json(build_result(cid))
                if p == "/api/delete":
                    shutil.rmtree(os.path.join(CACHE_DIR, cid), ignore_errors=True)
                    return self.send_json({"ok": True})
                if p == "/api/split":
                    hz = float(q["hz"]) if q.get("hz") else None
                    mode = q.get("mode", "auto") if q.get("mode") in ("auto", "duet", "female", "male") else "auto"
                    return self.send_json({"job": start_job(job_split, cid, mode, hz)})
                if p == "/api/lyrics":
                    return self.send_json({"job": start_job(job_lyrics, cid)})
                st = max(-12, min(12, int(q.get("st", "0"))))
                return self.send_json({"job": start_job(job_render, cid, st, q.get("which", "karaoke"))})
            if p in ("/", "", "/karaoke.html", "/index.html"):
                return self.send_file(os.path.join(ROOT, "karaoke.html"))
            if p.startswith("/cache/"):
                rel = os.path.normpath(urllib.parse.unquote(p[len("/cache/"):]))
                full = os.path.join(CACHE_DIR, rel)
                if not os.path.abspath(full).startswith(os.path.abspath(CACHE_DIR) + os.sep) or not re.search(r"\.(mp3|wav|json|lrc)$", full):
                    return self.send_json({"error": "Not found"}, 404)
                return self.send_file(full)
            return self.send_json({"error": "Not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self.send_json({"error": str(e), "code": "failed"}, 500)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == "/api/upload":
                if not (have("demucs") and shutil.which("ffmpeg")):
                    return self.send_json({"error": "Install the AI parts first: pip install demucs faster-whisper yt-dlp (and ffmpeg)", "code": "nodeps"}, 400)
                n = int(self.headers.get("Content-Length") or 0)
                os.makedirs(CACHE_DIR, exist_ok=True)
                tmp = os.path.join(CACHE_DIR, "up-%s.bin" % uuid.uuid4().hex[:8])
                h = hashlib.sha1()
                with open(tmp, "wb") as f:
                    left = n
                    while left > 0:
                        b = self.rfile.read(min(1 << 20, left))
                        if not b:
                            break
                        h.update(b)
                        f.write(b)
                        left -= len(b)
                if left > 0:
                    os.remove(tmp)
                    return self.send_json({"error": "The upload was cut off. Try again.", "code": "failed"}, 400)
                cid = h.hexdigest()[:16]
                d = cdir(cid)
                name = q.get("name", "song.mp3")
                ext = os.path.splitext(name)[1].lower()[:6] or ".mp3"
                if not any(x.startswith("src.") for x in os.listdir(d)):
                    shutil.move(tmp, os.path.join(d, "src" + ext))
                else:
                    os.remove(tmp)
                return self.send_json({"job": start_job(job_file, cid, q.get("lyrics") == "1", clean_title(name), "")})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if u.path == "/api/url":
                if not (have("demucs") and have("yt_dlp") and shutil.which("ffmpeg")):
                    return self.send_json({"error": "Install: pip install demucs faster-whisper yt-dlp (and ffmpeg)", "code": "nodeps"}, 400)
                return self.send_json({"job": start_job(job_url, body.get("url", ""), bool(body.get("lyrics")))})
            if u.path == "/api/remix":
                cid = body.get("id", "")
                if not valid_id(cid):
                    return self.send_json({"error": "Unknown song", "code": "nosong"}, 404)
                return self.send_json({"job": start_job(job_remix, cid, body.get("segments") or [], body.get("guide") or 0)})
            return self.send_json({"error": "Not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            return self.send_json({"error": str(e), "code": "failed"}, 500)


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def make_cert():
    d = os.path.join(CACHE_DIR, "_ssl")
    os.makedirs(d, exist_ok=True)
    crt, key = os.path.join(d, "cert.pem"), os.path.join(d, "key.pem")
    if not (os.path.exists(crt) and os.path.exists(key)):
        if not shutil.which("openssl"):
            raise SystemExit("--https needs the openssl command. Install it, or run without --https.")
        san = "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:%s" % lan_ip()
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", crt, "-days", "825",
                        "-subj", "/CN=karaoke", "-addext", san], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return crt, key


def main():
    ap = argparse.ArgumentParser()
    # On a host like Render/Railway/HF Spaces, PORT is injected and the app must bind 0.0.0.0.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default=("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"))
    ap.add_argument("--https", action="store_true", help="self-signed https so a phone may use the microphone")
    a = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(ROOT, "karaoke.html")):
        print("Warning: karaoke.html must sit next to server.py", file=sys.stderr)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.daemon_threads = True
    scheme = "http"
    if a.https:
        crt, key = make_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print("Smart Hindi Karaoke  ->  %s://localhost:%d   (Ctrl+C to stop)" % (scheme, a.port))
    if a.host in ("0.0.0.0", ""):
        print("  On your phone (same Wi-Fi):  %s://%s:%d%s" % (scheme, lan_ip(), a.port, "   (accept the certificate warning once)" if a.https else "   (the microphone needs --https)"))
    for label, ok in (("demucs (voice removal)", have("demucs")), ("faster-whisper (Hindi lyrics)", have("faster_whisper")),
                      ("yt-dlp (YouTube)", have("yt_dlp")), ("ffmpeg", bool(shutil.which("ffmpeg"))),
                      ("rubberband (best key change)", has_rubberband())):
        print("  [%s] %s" % ("x" if ok else " ", label))
    if not have("demucs"):
        print("  Missing parts? run:  pip install demucs faster-whisper yt-dlp")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
