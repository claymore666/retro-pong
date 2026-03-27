#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════╗
║         R E T R O   P O N G               ║
║   80s Arcade Style  ·  ML Opponent        ║
╚═══════════════════════════════════════════╝
"""

import curses
import locale
import time
import json
import math
import random
import subprocess
import threading
import io
import sys
import os
import wave as wave_mod
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("\033[1;31m")
    print("  ╔════════════════════════════════════╗")
    print("  ║  ERROR: NumPy is required!         ║")
    print("  ║  Run:  pip install numpy            ║")
    print("  ╚════════════════════════════════════╝")
    print("\033[0m")
    sys.exit(1)

locale.setlocale(locale.LC_ALL, '')


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

FPS = 30
WINNING_SCORE = 11
PADDLE_H = 4
BALL_SPD_INIT = 0.55
BALL_SPD_INC = 0.018
BALL_SPD_MAX = 1.5
PLAYER_SPD = 1.2
AI_SPD_BASE = 0.85
AI_SPD_INC = 0.01
MIN_W, MIN_H = 60, 24
SCORE_ROWS = 7          # rows for score display area

CH_BALL = '\u25cf'       # ●
CH_PAD = '\u2588'        # █
CH_TRAIL = ['\u25c9', '\u25cb', '\u00b7']  # ◉ ○ ·
CH_CENTER = '\u250a'     # ┊
PARTICLE_CH = ['*', '+', '\u00b7', '.', 'o']

# colour-pair ids
CP_NORM = 1;  CP_BRIGHT = 2;  CP_SCORE = 3;  CP_DANGER = 4
CP_INFO = 5;  CP_DIM = 6;     CP_PAD_P = 7;  CP_PAD_AI = 8
CP_BORDER = 9; CP_BALL = 10;  CP_TITLE = 11; CP_TRAIL = 12


# ═══════════════════════════════════════════════════════════════
#  FONTS
# ═══════════════════════════════════════════════════════════════

DIGITS = {
    '0': ['███', '█ █', '█ █', '█ █', '███'],
    '1': [' █ ', '██ ', ' █ ', ' █ ', '███'],
    '2': ['███', '  █', '███', '█  ', '███'],
    '3': ['███', '  █', '███', '  █', '███'],
    '4': ['█ █', '█ █', '███', '  █', '  █'],
    '5': ['███', '█  ', '███', '  █', '███'],
    '6': ['███', '█  ', '███', '█ █', '███'],
    '7': ['███', '  █', ' █ ', ' █ ', ' █ '],
    '8': ['███', '█ █', '███', '█ █', '███'],
    '9': ['███', '█ █', '███', '  █', '███'],
}

FONT = {
    'A': [' ███ ', '█   █', '█████', '█   █', '█   █'],
    'E': ['█████', '█    ', '███  ', '█    ', '█████'],
    'G': [' ████', '█    ', '█  ██', '█   █', ' ████'],
    'I': ['█████', '  █  ', '  █  ', '  █  ', '█████'],
    'M': ['█   █', '██ ██', '█ █ █', '█   █', '█   █'],
    'N': ['█   █', '██  █', '█ █ █', '█  ██', '█   █'],
    'O': [' ███ ', '█   █', '█   █', '█   █', ' ███ '],
    'P': ['████ ', '█   █', '████ ', '█    ', '█    '],
    'R': ['████ ', '█   █', '████ ', '█  █ ', '█   █'],
    'T': ['█████', '  █  ', '  █  ', '  █  ', '  █  '],
    'U': ['█   █', '█   █', '█   █', '█   █', ' ███ '],
    'V': ['█   █', '█   █', '█   █', ' █ █ ', '  █  '],
    'W': ['█   █', '█   █', '█ █ █', '██ ██', '█   █'],
    'Y': ['█   █', ' █ █ ', '  █  ', '  █  ', '  █  '],
    '!': ['  █  ', '  █  ', '  █  ', '     ', '  █  '],
    ' ': ['  ', '  ', '  ', '  ', '  '],
}


def text_art(text):
    """Render text as 5-row ASCII art."""
    rows = [''] * 5
    for ch in text.upper():
        g = FONT.get(ch, FONT[' '])
        for i in range(5):
            if rows[i]:
                rows[i] += ' '
            rows[i] += g[i]
    return rows


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def saddstr(win, y, x, s, attr=0):
    """Safe addstr – clips to window."""
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        if x < 0:
            s = s[-x:]
            x = 0
        s = s[:w - x]
        if s:
            win.addstr(y, x, s, attr)
    except curses.error:
        pass


def cx(total, text_w):
    return max(0, (total - text_w) // 2)


def predict_y(by, bdy, steps, fh):
    """Predict ball Y after *steps* frames, with wall bounces."""
    if fh <= 1 or steps <= 0:
        return by
    raw = by + bdy * steps
    period = 2.0 * (fh - 1)
    if period <= 0:
        return by
    y = raw % period
    if y < 0:
        y += period
    if y > fh - 1:
        y = period - y
    return max(0.0, min(float(fh - 1), y))


# ═══════════════════════════════════════════════════════════════
#  SOUND ENGINE  (chiptune via numpy + aplay)
# ═══════════════════════════════════════════════════════════════

class Sound:
    RATE = 22050

    def __init__(self):
        self.on = True
        self._ok = self._probe()
        self._music_proc = None
        self._fx = []

    # -- probing -------------------------------------------------------
    def _probe(self):
        try:
            return subprocess.run(
                ['aplay', '--version'], capture_output=True, timeout=2
            ).returncode == 0
        except Exception:
            return False

    # -- waveform generators -------------------------------------------
    def _sq(self, freq, dur, vol=0.15):
        t = np.linspace(0, dur, int(self.RATE * dur), False)
        w = np.sign(np.sin(2 * np.pi * freq * t))
        n = len(w)
        env = np.ones(n)
        a = min(int(0.005 * self.RATE), n)
        r = min(int(0.01 * self.RATE), n)
        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if r > 0:
            env[-r:] = np.linspace(1, 0, r)
        return (w * env * vol * 32767).astype(np.int16)

    def _tri(self, freq, dur, vol=0.20):
        t = np.linspace(0, dur, int(self.RATE * dur), False)
        w = 2.0 * np.abs(2.0 * (t * freq - np.floor(t * freq + 0.5))) - 1.0
        return (w * vol * 32767).astype(np.int16)

    def _noise(self, dur, vol=0.06):
        n = int(self.RATE * dur)
        return (np.random.uniform(-1, 1, n) * vol * 32767).astype(np.int16)

    # -- playback ------------------------------------------------------
    def _play(self, samples):
        if not self._ok or not self.on:
            return
        wav = self._to_wav(samples)
        try:
            p = subprocess.Popen(
                ['aplay', '-q'], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            def _feed():
                try:
                    p.stdin.write(wav)
                    p.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            threading.Thread(target=_feed, daemon=True).start()
            self._fx.append(p)
        except Exception:
            pass

    def _to_wav(self, samples):
        buf = io.BytesIO()
        with wave_mod.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(samples.tobytes())
        return buf.getvalue()

    # -- sound effects -------------------------------------------------
    def hit(self):
        self._play(self._sq(880, 0.05, 0.12))

    def wall(self):
        self._play(self._sq(440, 0.03, 0.08))

    def score(self):
        self._play(np.concatenate([
            self._sq(600, 0.08, 0.15),
            self._sq(400, 0.08, 0.15),
            self._sq(200, 0.16, 0.15),
        ]))

    def win(self):
        self._play(np.concatenate([
            self._sq(f, 0.10, 0.18)
            for f in [523, 587, 659, 784, 880, 1047]
        ]))

    def lose(self):
        self._play(np.concatenate([
            self._sq(f, 0.13, 0.18) for f in [400, 350, 300, 250, 200]
        ]))

    # -- background music ----------------------------------------------
    def start_music(self):
        if not self._ok or not self.on:
            return
        self.stop_music()
        raw = self._make_music().tobytes()
        try:
            self._music_proc = subprocess.Popen(
                ['aplay', '-f', 'S16_LE', '-r', str(self.RATE),
                 '-c', '1', '-q'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            def _loop():
                try:
                    while self._music_proc and self._music_proc.poll() is None:
                        self._music_proc.stdin.write(raw)
                        self._music_proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            threading.Thread(target=_loop, daemon=True).start()
        except Exception:
            self._music_proc = None

    def stop_music(self):
        if self._music_proc:
            try:
                self._music_proc.kill()
                self._music_proc.wait(timeout=1)
            except Exception:
                pass
            self._music_proc = None

    def toggle(self):
        self.on = not self.on
        if self.on:
            self.start_music()
        else:
            self.stop_music()

    def cleanup(self):
        self.stop_music()
        for p in self._fx:
            try:
                p.kill()
            except Exception:
                pass

    def _make_music(self):
        """Generate a full chiptune track with verse / chorus / bridge
        structure (~55 seconds) before looping.

        Layout
        ──────
          Intro (4 bars)  – bass + hihat only, builds tension
          Verse A (8 bars) – classic Cm arpeggio theme
          Verse A' (8 bars) – same chords, descending melody variation
          Chorus (8 bars)  – higher energy, Fm→G→Ab→Bb progression
          Bridge (8 bars)  – halftime feel, sustained notes, Eb→Fm→G
          Verse A (8 bars) – theme returns
          Outro (4 bars)   – melody fades, bass walks back to root
        """
        bpm = 140
        n8 = 60.0 / bpm / 2          # eighth-note duration (≈0.214s)
        s8 = int(self.RATE * n8)      # samples per eighth

        # ── note frequencies ──────────────────────────────────
        # fmt: off
        C2=65;   D2=73;   Eb2=78;  F2=87;   G2=98
        Ab2=104; Bb2=117; B2=123;  C3=131;  D3=147
        Eb3=156; F3=175;  G3=196;  Ab3=208; Bb3=233
        B3=247;  C4=262;  D4=294;  Eb4=311; F4=349
        G4=392;  Ab4=415; Bb4=466; B4=494;  C5=523
        D5=587;  Eb5=659; F5=698;  G5=784
        REST = 0
        # fmt: on

        # Helper: render a section from (melody, bass, hihat_pattern)
        # Each melody/bass entry is one eighth note.
        def _render(mel, bass_notes, hat_pat, lead_vol=0.10,
                    bass_vol=0.18, hat_vol=0.04, note_len=0.85):
            n = len(mel)
            total = s8 * n
            lead = np.zeros(total, dtype=np.int16)
            bass = np.zeros(total, dtype=np.int16)
            hh   = np.zeros(total, dtype=np.int16)

            for i, f in enumerate(mel):
                off = i * s8
                if f > 0:
                    s = self._sq(f, n8 * note_len, lead_vol)
                    e = min(off + len(s), total)
                    lead[off:e] = s[:e - off]

            # bass: one entry per half-note (4 eighths)
            for i, f in enumerate(bass_notes):
                off = i * s8 * 4
                if f > 0:
                    s = self._tri(f, n8 * 3.9, bass_vol)
                    e = min(off + len(s), total)
                    bass[off:e] = s[:e - off]

            # hihat pattern repeats every bar (8 eighths)
            for i in range(n):
                pi = i % len(hat_pat)
                if hat_pat[pi]:
                    off = i * s8
                    s = self._noise(n8 * 0.25, hat_vol)
                    e = min(off + len(s), total)
                    hh[off:e] = s[:e - off]

            return np.clip(
                lead.astype(np.int32) + bass.astype(np.int32)
                + hh.astype(np.int32), -32767, 32767
            ).astype(np.int16)

        # ── INTRO (4 bars = 32 eighths) ──────────────────────
        # No melody, just bass and hihat building up
        intro_mel  = [REST]*32
        intro_bass = [C2, C2, C2, C2, Ab2, Ab2, G2, G2]
        intro_hat  = [1,0,0,0, 1,0,0,0]  # sparse kick pattern
        intro = _render(intro_mel, intro_bass, intro_hat,
                        hat_vol=0.05)

        # ── VERSE A (8 bars = 64 eighths) ────────────────────
        # Classic ascending Cm arpeggios – the main theme
        va_mel = [
            # Cm (2 bars)
            C4, Eb4, G4, C5, G4, Eb4, C4, Eb4,
            C4, G4,  C5, G4, Eb4, C4, Eb4, G4,
            # Bb (2 bars)
            Bb3, D4, F4, Bb4, F4, D4, Bb3, D4,
            Bb3, F4, Bb4, F4, D4, Bb3, D4, F4,
            # Ab (2 bars)
            Ab3, C4, Eb4, Ab4, Eb4, C4, Ab3, C4,
            Ab3, Eb4, Ab4, Eb4, C4, Ab3, C4, Eb4,
            # G → Cm (2 bars)
            G3, B3, D4, G4, D4, B3, G3, B3,
            G3, D4, G4, B4, G4, D4, C4, Eb4,
        ]
        #                   16 entries = 16 half-notes = 8 bars
        va_bass = [C2, C2, C2, C3, Bb2, Bb2, Bb2, Bb2,
                   Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C2]
        va_hat  = [1,0,1,0, 1,0,1,0]
        verse_a = _render(va_mel, va_bass, va_hat)

        # ── VERSE A' (8 bars) ────────────────────────────────
        # Subtle variation: same chords, melody goes down then up
        # (only small changes from verse A – a few notes swapped)
        va2_mel = [
            # Cm – starts high, comes back (subtle inversion)
            C5, G4, Eb4, C4, Eb4, G4, C5, G4,
            C4, G4,  C5, G4, Eb4, C4, Eb4, G4,
            # Bb – same shape, starts from top
            Bb4, F4, D4, Bb3, D4, F4, Bb4, F4,
            Bb3, F4, Bb4, F4, D4, Bb3, D4, F4,
            # Ab – identical to verse A (anchor)
            Ab3, C4, Eb4, Ab4, Eb4, C4, Ab3, C4,
            Ab3, Eb4, Ab4, Eb4, C4, Ab3, C4, Eb4,
            # G → Cm – slight embellishment at end
            G3, B3, D4, G4, D4, B3, G3, B3,
            G3, D4, G4, B4, G4, Eb4, C4, G4,
        ]
        va2_bass = [C2, C2, C2, C3, Bb2, Bb2, Bb2, Bb2,
                    Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C2]
        va2_hat  = [1,0,1,0, 1,0,1,1]
        verse_a2 = _render(va2_mel, va2_bass, va2_hat)

        # ── CHORUS (8 bars) ──────────────────────────────────
        # Same Cm→Bb→Ab→G chords but melody peaks one octave higher
        # at phrase endings – subtle lift, not a different song
        ch_mel = [
            # Cm – same arpeggio, but bar 2 peaks at C5
            C4, Eb4, G4, C5, G4, Eb4, C4, G4,
            C4, Eb4, G4, C5, Eb5, C5, G4, Eb4,
            # Bb – same shape, bar 2 peaks at Bb4
            Bb3, D4, F4, Bb4, F4, D4, Bb3, F4,
            Bb3, D4, F4, Bb4, D5, Bb4, F4, D4,
            # Ab – same, bar 2 peaks at Ab4→C5
            Ab3, C4, Eb4, Ab4, Eb4, C4, Ab3, Eb4,
            Ab3, C4, Eb4, Ab4, C5, Ab4, Eb4, C4,
            # G → Cm – builds to highest note then resolves
            G3, B3, D4, G4, D4, B3, G4, B4,
            G4, B4, D5, G5, D5, B4, C5, Eb5,
        ]
        ch_bass = [C2, C2, C2, C3, Bb2, Bb2, Bb2, Bb2,
                   Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C3]
        ch_hat  = [1,0,1,1, 1,0,1,1]  # slightly busier hihat
        chorus = _render(ch_mel, ch_bass, ch_hat,
                         lead_vol=0.12, hat_vol=0.05)

        # ── BRIDGE (8 bars) ──────────────────────────────────
        # Same chords, halftime feel – notes held longer (repeated)
        # Gives breathing room; still recognisably the same tune
        br_mel = [
            # Cm – long tones
            C4, C4, Eb4, Eb4, G4, G4, C5, C5,
            G4, G4, Eb4, Eb4, C4, C4, C4, C4,
            # Bb – long tones
            Bb3, Bb3, D4, D4, F4, F4, Bb4, Bb4,
            F4, F4, D4, D4, Bb3, Bb3, Bb3, Bb3,
            # Ab – long tones
            Ab3, Ab3, C4, C4, Eb4, Eb4, Ab4, Ab4,
            Eb4, Eb4, C4, C4, Ab3, Ab3, Ab3, Ab3,
            # G → Cm – builds back into the theme
            G3, G3, B3, B3, D4, D4, G4, G4,
            G4, G4, B4, B4, D4, D4, C4, Eb4,
        ]
        br_bass = [C2, C2, C2, C2, Bb2, Bb2, Bb2, Bb2,
                   Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C2]
        br_hat  = [1,0,0,0, 1,0,0,0]  # sparse – halftime feel
        bridge = _render(br_mel, br_bass, br_hat,
                         lead_vol=0.09, bass_vol=0.20,
                         note_len=0.95)

        # ── VERSE A reprise (8 bars) – same as verse_a ───────
        reprise = _render(va_mel, va_bass, va_hat)

        # ── OUTRO (4 bars = 32 eighths) ──────────────────────
        # Melody thins out, bass walks to root
        ou_mel = [
            C5, REST, G4, REST, Eb4, REST, C4, REST,
            G4, REST, Eb4, REST, C4, REST, REST, REST,
            Eb4, REST, C4, REST, G3, REST, REST, REST,
            C4, REST, REST, REST, REST, REST, REST, REST,
        ]
        ou_bass = [C2, Ab2, F2, Eb2, Ab2, G2, G2, C2]
        ou_hat  = [1,0,0,0, 0,0,0,0]
        outro = _render(ou_mel, ou_bass, ou_hat,
                        lead_vol=0.08, hat_vol=0.03)

        # ── assemble full track ───────────────────────────────
        return np.concatenate([
            intro, verse_a, verse_a2, chorus,
            bridge, reprise, outro,
        ])


# ═══════════════════════════════════════════════════════════════
#  NEURAL NETWORK  (2-layer MLP)
# ═══════════════════════════════════════════════════════════════

class NN:
    def __init__(self, ni=6, nh=64, no=3, lr=0.002):
        self.W1 = np.random.randn(ni, nh) * np.sqrt(2.0 / ni)
        self.b1 = np.zeros(nh)
        self.W2 = np.random.randn(nh, no) * np.sqrt(2.0 / nh)
        self.b2 = np.zeros(no)
        self.lr = lr

    def forward(self, X):
        X = np.atleast_2d(X)
        self._X = X
        self._Z1 = X @ self.W1 + self.b1
        self._A1 = np.maximum(0, self._Z1)
        return self._A1 @ self.W2 + self.b2

    def train(self, X, T):
        X = np.atleast_2d(X)
        T = np.atleast_2d(T)
        P = self.forward(X)
        d = P - T
        bs = X.shape[0]
        dZ2 = d / bs
        dW2 = self._A1.T @ dZ2
        db2 = dZ2.sum(0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self._Z1 > 0)
        dW1 = self._X.T @ dZ1
        db1 = dZ1.sum(0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return float(np.mean(d ** 2))


# ═══════════════════════════════════════════════════════════════
#  AI OPPONENT  (DQN with pre-training)
# ═══════════════════════════════════════════════════════════════

class AI:
    """AI with human-like progression.

    Lvl 1-2: tracks ball's current Y (beginner, no prediction)
    Lvl 3-4: crude linear extrapolation, no bounce awareness
    Lvl 5-7: learns to handle 1 bounce, still noisy
    Lvl 8+:  near-perfect trajectory prediction (earned)

    Each level also increases movement speed and reduces positional noise.
    The neural network trains on live experience and gradually supplements
    the rule-based behaviour.
    """
    ACTIONS = [-1, 0, 1]   # up / stay / down

    def __init__(self, fw, fh):
        self.fw, self.fh = fw, fh
        self.nn = NN(6, 128, 3)
        self.buf = deque(maxlen=5000)
        self.eps = 0.04
        self.gamma = 0.9
        self.hits = 0
        self.level = 1
        self.last_s = None
        self.last_a = None
        self.nn_mix = 0.0       # 0 = pure rule, 1 = pure NN
        self._pretrain()

    def _state(self, bx, by, bdx, bdy, acy):
        return np.array([
            bx / max(self.fw, 1),
            by / max(self.fh, 1),
            bdx / BALL_SPD_MAX,
            bdy / BALL_SPD_MAX,
            acy / max(self.fh, 1),
            max(0, self.fw - bx) / max(self.fw, 1),
        ])

    # ── level-aware heuristic ─────────────────────────────────
    def _rule(self, bx, by, bdx, bdy, acy):
        """Heuristic whose skill depends on self.level.

        Lvl 1-2: lazy, only reacts when ball is close, tracks current Y
        Lvl 3-4: starts predicting linearly, still noisy & slow to react
        Lvl 5-7: bounce-aware prediction, moderate noise
        Lvl 8+ : near-perfect prediction, fast reactions
        """
        lvl = self.level

        if bdx <= 0.01:
            # ball heading away → drift toward centre (lazy)
            if lvl < 5:
                return 1        # low-level AI just stands still
            ty = self.fh / 2.0
            diff = ty - acy
            return 0 if diff < -2 else (2 if diff > 2 else 1)

        # --- attention: low-level AI ignores ball until it's close ---
        #   lvl 1 → reacts when ball past 55% of field width
        #   lvl 5 → reacts when past 15%
        #   lvl 8+ → always attentive
        react_x = self.fw * max(0.0, 0.65 - lvl * 0.08)
        if bx < react_x:
            return 1            # "I'll worry about it later"

        # --- hesitation: sometimes freezes instead of moving ---
        #   lvl 1 → 10%,  lvl 5 → 2%,  lvl 8+ → 0%
        hesitate = max(0.0, 0.12 - lvl * 0.015)
        if random.random() < hesitate:
            return 1

        steps = max(1, (self.fw - 3 - bx) / bdx)

        # --- prediction quality depends on level ---
        if lvl <= 2:
            # Beginner: just track ball's current Y position
            ty = by
        elif lvl <= 4:
            # Novice: linear extrapolation, no bounce awareness
            ty = by + bdy * steps
            ty = max(0.0, min(float(self.fh - 1), ty))
        elif lvl <= 7:
            # Intermediate: bounce-aware prediction
            ty = predict_y(by, bdy, steps, self.fh)
        else:
            # Expert: full prediction
            ty = predict_y(by, bdy, steps, self.fh)

        # --- positional noise (shrinks with level) ---
        #   lvl 1 → sigma ≈ 3.5  (noticeably off)
        #   lvl 5 → sigma ≈ 1.5
        #   lvl 10 → sigma ≈ 0.3
        noise = max(0.3, 4.0 - lvl * 0.4)
        ty += random.gauss(0, noise)

        # --- at low levels, blend toward ball's current Y ---
        #   lvl 1 → 50% current Y, 50% predicted
        #   lvl 5 → 10% current Y
        #   lvl 8+ → 0%
        track = max(0.0, 0.6 - lvl * 0.08)
        ty = ty * (1.0 - track) + by * track

        ty = max(0.0, min(float(self.fh - 1), ty))

        # --- dead-zone: bigger at low levels ---
        #   lvl 1 → 2.2   (tolerates small offsets)
        #   lvl 5 → 1.2
        #   lvl 10 → 0.6
        deadzone = max(0.6, 2.5 - lvl * 0.2)
        diff = ty - acy
        if diff < -deadzone:
            return 0
        if diff > deadzone:
            return 2
        return 1

    # ── pre-training ──────────────────────────────────────────
    def _pretrain(self):
        """Pre-train NN on demonstrations at *current* skill level."""
        S, T = [], []
        for _ in range(5000):
            bx = random.uniform(0, self.fw)
            by = random.uniform(1, self.fh - 1)
            spd = random.uniform(BALL_SPD_INIT, BALL_SPD_MAX)
            ang = random.uniform(-math.pi / 3, math.pi / 3)
            bdx = abs(spd * math.cos(ang))
            bdy = spd * math.sin(ang)
            acy = random.uniform(PADDLE_H / 2, self.fh - PADDLE_H / 2)
            opt = self._rule(bx, by, bdx, bdy, acy)
            s = self._state(bx, by, bdx, bdy, acy)
            t = np.array([-1.0, -1.0, -1.0])
            t[opt] = 1.0
            S.append(s)
            T.append(t)

        S, T = np.array(S), np.array(T)
        for _ in range(100):
            idx = np.random.permutation(len(S))
            for i in range(0, len(S), 64):
                b = idx[i:i + 64]
                self.nn.train(S[b], T[b])

    # ── action selection ──────────────────────────────────────
    def act(self, bx, by, bdx, bdy, acy):
        s = self._state(bx, by, bdx, bdy, acy)
        if random.random() < self.eps:
            a = random.randrange(3)
        elif random.random() < self.nn_mix:
            a = int(np.argmax(self.nn.forward(s)[0]))
        else:
            a = self._rule(bx, by, bdx, bdy, acy)
        self.last_s, self.last_a = s, a
        return a

    def record(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def on_hit(self):
        self.hits += 1
        self.level = 1 + self.hits // 3
        self.eps = max(0.01, self.eps * 0.95)
        self.nn_mix = min(0.85, self.nn_mix + 0.03)
        if self.last_s is not None:
            self.record(self.last_s, self.last_a, 1.0, self.last_s, True)
        self._learn()

    def on_miss(self):
        if self.last_s is not None:
            self.record(self.last_s, self.last_a, -1.0, self.last_s, True)
        self._learn()

    def _learn(self):
        if len(self.buf) < 32:
            return
        batch = random.sample(list(self.buf), min(64, len(self.buf)))
        Sb = np.array([t[0] for t in batch])
        Ab = [t[1] for t in batch]
        Rb = [t[2] for t in batch]
        Nb = np.array([t[3] for t in batch])
        Db = [t[4] for t in batch]
        cq = self.nn.forward(Sb).copy()
        nq = self.nn.forward(Nb)
        tgt = cq.copy()
        for i in range(len(batch)):
            if Db[i]:
                tgt[i, Ab[i]] = Rb[i]
            else:
                tgt[i, Ab[i]] = Rb[i] + self.gamma * np.max(nq[i])
        self.nn.train(Sb, tgt)

    @property
    def speed(self):
        # Lvl 1 → 0.65,  Lvl 5 → 0.85,  Lvl 10 → 1.1
        return min(1.2, 0.60 + self.level * 0.05)


# ═══════════════════════════════════════════════════════════════
#  PARTICLES & EFFECTS
# ═══════════════════════════════════════════════════════════════

class Particle:
    __slots__ = ('x', 'y', 'dx', 'dy', 'ch', 'life', 'mlife', 'cp')
    def __init__(self, x, y, dx, dy, ch='*', life=1.0, cp=CP_BRIGHT):
        self.x = float(x); self.y = float(y)
        self.dx = dx; self.dy = dy
        self.ch = ch; self.life = life; self.mlife = life; self.cp = cp


class FX:
    def __init__(self):
        self.parts = []
        self.shake = 0
        self.flash = 0
        self.msgs = []          # (text, y, x, ttl, cp)

    def explode(self, x, y, n=14):
        for _ in range(n):
            a = random.uniform(0, 2 * math.pi)
            v = random.uniform(0.3, 1.5)
            self.parts.append(Particle(
                x, y, math.cos(a) * v, math.sin(a) * v,
                random.choice(PARTICLE_CH),
                random.uniform(0.5, 1.5),
                random.choice([CP_SCORE, CP_DANGER, CP_BRIGHT, CP_INFO]),
            ))

    def popup(self, text, y, x, ttl=40, cp=CP_BRIGHT):
        self.msgs.append((text, y, x, ttl, cp))

    def update(self):
        alive = []
        for p in self.parts:
            p.x += p.dx; p.y += p.dy
            p.dy += 0.05
            p.life -= 1.0 / FPS
            if p.life > 0:
                alive.append(p)
        self.parts = alive
        if self.shake > 0:
            self.shake -= 1
        if self.flash > 0:
            self.flash -= 1
        self.msgs = [(t, y, x, ttl - 1, c)
                     for t, y, x, ttl, c in self.msgs if ttl > 1]

    def render(self, win):
        for p in self.parts:
            ry, rx = int(p.y), int(p.x)
            a = p.life / p.mlife
            ch = p.ch if a > 0.5 else ('.' if a > 0.2 else ' ')
            saddstr(win, ry, rx, ch, curses.color_pair(p.cp))
        for t, y, x, ttl, c in self.msgs:
            saddstr(win, y, x, t, curses.color_pair(c) | curses.A_BOLD)


# ═══════════════════════════════════════════════════════════════
#  HIGH-SCORE TABLE
# ═══════════════════════════════════════════════════════════════

class HiScores:
    FILE = Path.home() / '.retropong_scores.json'
    MAX = 10

    def __init__(self):
        self.data = self._load()

    def _load(self):
        try:
            return json.loads(self.FILE.read_text())[:self.MAX]
        except Exception:
            return []

    def _save(self):
        try:
            self.FILE.write_text(json.dumps(self.data, indent=2))
        except Exception:
            pass

    def qualifies(self, score):
        return len(self.data) < self.MAX or score > min(
            (e['score'] for e in self.data), default=0)

    def add(self, initials, score, ai_lvl):
        e = {'initials': initials, 'score': score,
             'ai_level': ai_lvl,
             'date': datetime.now().strftime('%Y-%m-%d')}
        self.data.append(e)
        self.data.sort(key=lambda x: -x['score'])
        self.data = self.data[:self.MAX]
        self._save()
        return self.data.index(e)

    def enter_initials(self, win):
        h, w = win.getmaxyx()
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        sel = [0, 0, 0]
        pos = 0
        win.timeout(120)
        while True:
            cy, cxv = h // 2, w // 2
            for row in range(cy - 6, cy + 8):
                saddstr(win, row, 0, ' ' * w)

            saddstr(win, cy - 4, cx(w, 19), 'ENTER YOUR INITIALS',
                    curses.color_pair(CP_SCORE) | curses.A_BOLD)
            for i in range(3):
                ch = letters[sel[i]]
                x = cxv - 5 + i * 4
                attr = curses.color_pair(CP_BRIGHT) | curses.A_BOLD
                if i == pos and int(time.time() * 4) % 2:
                    attr |= curses.A_REVERSE
                if i == pos:
                    saddstr(win, cy - 1, x, '\u25b2',
                            curses.color_pair(CP_INFO))
                    saddstr(win, cy + 1, x, '\u25bc',
                            curses.color_pair(CP_INFO))
                saddstr(win, cy, x, ch, attr)
                saddstr(win, cy + 2, x, '\u2500',
                        curses.color_pair(CP_DIM) | curses.A_BOLD)
            hint = 'UP/DOWN:SELECT  RIGHT/ENTER:NEXT'
            saddstr(win, cy + 4, cx(w, len(hint)), hint,
                    curses.color_pair(CP_DIM) | curses.A_BOLD)
            win.refresh()
            k = win.getch()
            if k in (curses.KEY_UP, ord('w'), ord('W')):
                sel[pos] = (sel[pos] - 1) % len(letters)
            elif k in (curses.KEY_DOWN, ord('s'), ord('S')):
                sel[pos] = (sel[pos] + 1) % len(letters)
            elif k in (curses.KEY_RIGHT, 9, 10, 13, curses.KEY_ENTER):
                if pos < 2:
                    pos += 1
                else:
                    break
            elif k == curses.KEY_LEFT and pos > 0:
                pos -= 1
        return ''.join(letters[i] for i in sel)

    def render(self, win, hl=-1):
        h, w = win.getmaxyx()
        cy = h // 2
        title = '\u2550\u2550\u2550 HIGH SCORES \u2550\u2550\u2550'
        saddstr(win, cy - 8, cx(w, len(title)), title,
                curses.color_pair(CP_SCORE) | curses.A_BOLD)
        hdr = f"{'#':>2}  {'NAME':<4} {'SCORE':>6}  {'LVL':>3}  {'DATE':<10}"
        saddstr(win, cy - 6, cx(w, len(hdr)), hdr,
                curses.color_pair(CP_INFO))
        saddstr(win, cy - 5, cx(w, len(hdr)), '\u2500' * len(hdr),
                curses.color_pair(CP_DIM) | curses.A_BOLD)
        for i, e in enumerate(self.data):
            ln = (f"{i+1:>2}. {e['initials']:<4} {e['score']:>6}  "
                  f"{e.get('ai_level','?'):>3}  {e.get('date',''):>10}")
            attr = curses.color_pair(CP_NORM)
            if i == hl:
                attr = curses.color_pair(CP_SCORE) | curses.A_BOLD
                if int(time.time() * 3) % 2:
                    attr |= curses.A_REVERSE
            saddstr(win, cy - 4 + i, cx(w, len(ln)), ln, attr)
        if not self.data:
            m = 'NO SCORES YET!'
            saddstr(win, cy, cx(w, len(m)), m,
                    curses.color_pair(CP_DIM) | curses.A_BOLD)


# ═══════════════════════════════════════════════════════════════
#  MAIN GAME
# ═══════════════════════════════════════════════════════════════

class RetroPong:

    def __init__(self, scr):
        self.scr = scr
        self.snd = Sound()
        self.fx = FX()
        self.hs = HiScores()
        self._init_curses()

    # ── curses setup ──────────────────────────────────────────
    def _init_curses(self):
        curses.curs_set(0)
        self.scr.nodelay(True)
        self.scr.keypad(True)
        curses.start_color()
        curses.use_default_colors()
        pairs = [
            (CP_NORM,   curses.COLOR_GREEN,   -1),
            (CP_BRIGHT, curses.COLOR_WHITE,   -1),
            (CP_SCORE,  curses.COLOR_YELLOW,  -1),
            (CP_DANGER, curses.COLOR_RED,     -1),
            (CP_INFO,   curses.COLOR_CYAN,    -1),
            (CP_DIM,    curses.COLOR_BLACK,   -1),
            (CP_PAD_P,  curses.COLOR_GREEN,   -1),
            (CP_PAD_AI, curses.COLOR_RED,     -1),
            (CP_BORDER, curses.COLOR_GREEN,   -1),
            (CP_BALL,   curses.COLOR_WHITE,   -1),
            (CP_TITLE,  curses.COLOR_MAGENTA, -1),
            (CP_TRAIL,  curses.COLOR_GREEN,   -1),
        ]
        for pid, fg, bg in pairs:
            try:
                curses.init_pair(pid, fg, bg)
            except curses.error:
                pass

    # ── size check ────────────────────────────────────────────
    def _ok_size(self):
        h, w = self.scr.getmaxyx()
        if w < MIN_W or h < MIN_H:
            self.scr.erase()
            m = f'Terminal too small! Need {MIN_W}x{MIN_H}, got {w}x{h}'
            saddstr(self.scr, h // 2, cx(w, len(m)), m,
                    curses.color_pair(CP_DANGER))
            self.scr.refresh()
            self.scr.timeout(-1)
            self.scr.getch()
            return False
        return True

    # ── intro sequence ────────────────────────────────────────
    def _intro(self):
        if not self._ok_size():
            return False
        h, w = self.scr.getmaxyx()
        self.scr.timeout(50)

        stars = [(random.randint(0, w - 1), random.randint(0, h - 1),
                  random.uniform(0.5, 2.0)) for _ in range(50)]

        def _draw_stars():
            for i, (sx, sy, sp) in enumerate(stars):
                sx -= sp
                if sx < 0:
                    sx = w - 1
                    sy = random.randint(0, h - 1)
                stars[i] = (sx, sy, sp)
                ch = '.' if sp < 1.0 else ('*' if sp < 1.5 else '#')
                saddstr(self.scr, int(sy), int(sx), ch,
                        curses.color_pair(CP_DIM) | curses.A_BOLD)

        # Phase 1 – starfield (2 s)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            self.scr.erase()
            _draw_stars()
            self.scr.refresh()
            if self.scr.getch() != -1:
                return True

        # Phase 2 – logo reveal
        logo_retro = text_art('RETRO')
        logo_pong = text_art('PONG')
        logo = logo_retro + [''] + logo_pong
        ly = max(1, (h - len(logo) - 10) // 2)

        chars = []
        for ri, line in enumerate(logo):
            lx = cx(w, len(line))
            for ci, ch in enumerate(line):
                if ch.strip():
                    chars.append((ly + ri, lx + ci, ch))
        random.shuffle(chars)
        batch = max(1, len(chars) // 15)

        revealed = []
        idx = 0
        while idx < len(chars):
            self.scr.erase()
            _draw_stars()
            revealed.extend(chars[idx:idx + batch])
            for ry, rx, ch in revealed:
                saddstr(self.scr, ry, rx, ch,
                        curses.color_pair(CP_TITLE) | curses.A_BOLD)
            idx += batch
            self.scr.refresh()
            if self.scr.getch() != -1:
                return True
            time.sleep(0.04)

        # Phase 3 – taglines + INSERT COIN
        sub = '\u2500\u2500 80s ARCADE  \u00b7  ML OPPONENT \u2500\u2500'
        sub_y = ly + len(logo) + 1
        saddstr(self.scr, sub_y, cx(w, len(sub)), sub,
                curses.color_pair(CP_INFO))
        cred = 'POWERED BY MACHINE LEARNING'
        saddstr(self.scr, sub_y + 2, cx(w, len(cred)), cred,
                curses.color_pair(CP_DIM) | curses.A_BOLD)

        coin = '>>> INSERT COIN <<<'
        prompt = 'PRESS ANY KEY TO START'
        t0 = time.monotonic()
        while True:
            dt = time.monotonic() - t0
            if int(dt * 3) % 2:
                saddstr(self.scr, sub_y + 5, cx(w, len(coin)), coin,
                        curses.color_pair(CP_SCORE) | curses.A_BOLD)
            else:
                saddstr(self.scr, sub_y + 5, cx(w, len(coin)),
                        ' ' * len(coin))
            if dt > 1.0 and int(dt * 2) % 2:
                saddstr(self.scr, sub_y + 7, cx(w, len(prompt)), prompt,
                        curses.color_pair(CP_BRIGHT))
            else:
                saddstr(self.scr, sub_y + 7, cx(w, len(prompt)),
                        ' ' * len(prompt))
            self.scr.refresh()
            if self.scr.getch() != -1:
                return True
            if dt > 30:
                return True
        return True

    # ── field init ────────────────────────────────────────────
    def _init_field(self):
        h, w = self.scr.getmaxyx()
        self.th, self.tw = h, w
        self.f_top = SCORE_ROWS + 1
        self.f_left = 1
        self.fw = w - 2
        self.fh = h - SCORE_ROWS - 3

        self.px_p = 2
        self.px_ai = self.fw - 3
        self.py_p = self.fh / 2.0 - PADDLE_H / 2
        self.py_ai = self.fh / 2.0 - PADDLE_H / 2

        self.bx = self.fw / 2.0
        self.by = self.fh / 2.0
        self.bdx = 0.0
        self.bdy = 0.0
        self.bspd = BALL_SPD_INIT
        self.rally = 0
        self.trail = deque(maxlen=3)

        self.sc_p = 0
        self.sc_ai = 0
        self.ai = AI(self.fw, self.fh)
        self.paused = False
        self.rnd = 1
        self.fx = FX()
        self._serve()

    def _serve(self, d=None):
        self.bx = self.fw / 2.0
        self.by = self.fh / 2.0
        d = d or random.choice([-1, 1])
        a = random.uniform(-math.pi / 4, math.pi / 4)
        self.bdx = math.cos(a) * d
        self.bdy = math.sin(a)
        m = math.hypot(self.bdx, self.bdy)
        if m:
            self.bdx /= m
            self.bdy /= m
        self.bspd = min(BALL_SPD_INIT + BALL_SPD_INC * self.rally,
                        BALL_SPD_MAX)
        self.trail.clear()

    # ── physics ───────────────────────────────────────────────
    def _step_ball(self):
        self.trail.append((self.bx, self.by))
        self.bx += self.bdx * self.bspd
        self.by += self.bdy * self.bspd

        # wall bounce
        if self.by < 0:
            self.by = -self.by
            self.bdy = abs(self.bdy)
            self.snd.wall()
        elif self.by >= self.fh - 1:
            self.by = 2 * (self.fh - 1) - self.by
            self.bdy = -abs(self.bdy)
            self.snd.wall()

        # player paddle
        if (self.bdx < 0 and
                self.px_p - 0.5 <= self.bx <= self.px_p + 1.5 and
                self.py_p - 0.5 <= self.by <= self.py_p + PADDLE_H - 0.5):
            self.bdx = abs(self.bdx)
            hp = (self.by - self.py_p) / PADDLE_H
            self.bdy = (hp - 0.5) * 2.0
            m = math.hypot(self.bdx, self.bdy)
            if m:
                self.bdx /= m; self.bdy /= m
            self.bx = self.px_p + 1.6
            self.rally += 1
            self.bspd = min(BALL_SPD_INIT + BALL_SPD_INC * self.rally,
                            BALL_SPD_MAX)
            self.snd.hit()

        # AI paddle
        elif (self.bdx > 0 and
              self.px_ai - 1.5 <= self.bx <= self.px_ai + 0.5 and
              self.py_ai - 0.5 <= self.by <= self.py_ai + PADDLE_H - 0.5):
            self.bdx = -abs(self.bdx)
            hp = (self.by - self.py_ai) / PADDLE_H
            self.bdy = (hp - 0.5) * 2.0
            m = math.hypot(self.bdx, self.bdy)
            if m:
                self.bdx /= m; self.bdy /= m
            self.bx = self.px_ai - 1.6
            self.rally += 1
            self.bspd = min(BALL_SPD_INIT + BALL_SPD_INC * self.rally,
                            BALL_SPD_MAX)
            self.ai.on_hit()
            self.snd.hit()
            if self.ai.hits > 0 and self.ai.hits % 3 == 0:
                self.fx.popup(
                    f'AI EVOLVED! LVL {self.ai.level}',
                    self.f_top + 1,
                    cx(self.tw, 20), 45, CP_DANGER)

        # scoring
        scored = None
        if self.bx < -1:
            scored = 'ai'; self.sc_ai += 1
        elif self.bx > self.fw:
            scored = 'player'; self.sc_p += 1

        if scored:
            self.snd.score()
            ex = 3 if scored == 'ai' else self.fw - 3
            self.fx.explode(self.f_left + ex,
                            self.f_top + int(self.by), 18)
            self.fx.flash = 5
            self.fx.shake = 6
            if scored == 'ai':
                self.ai.on_miss()
            self._draw()
            self.scr.refresh()
            time.sleep(0.6)
            self.rally = 0
            self.rnd += 1
            self._serve(1 if scored == 'player' else -1)
            if self.sc_p < WINNING_SCORE and self.sc_ai < WINNING_SCORE:
                self._show_round()
        return scored

    # ── AI update ─────────────────────────────────────────────
    def _step_ai(self):
        acy = self.py_ai + PADDLE_H / 2
        a = self.ai.act(self.bx, self.by,
                        self.bdx * self.bspd, self.bdy * self.bspd, acy)
        self.py_ai += self.ai.ACTIONS[a] * self.ai.speed
        self.py_ai = max(0, min(self.fh - PADDLE_H, self.py_ai))

        # record experience
        ncy = self.py_ai + PADDLE_H / 2
        ns = self.ai._state(self.bx, self.by,
                            self.bdx * self.bspd, self.bdy * self.bspd, ncy)
        if self.bdx > 0 and self.bspd > 0:
            steps = max(0, (self.px_ai - self.bx) /
                        max(0.01, self.bdx * self.bspd))
            py = predict_y(self.by, self.bdy * self.bspd, steps, self.fh)
            od = abs(py - acy)
            nd = abs(py - ncy)
            r = 0.05 if nd < od else -0.02
        else:
            r = 0.0
        self.ai.record(self.ai.last_s, self.ai.last_a, r, ns, False)

    # ── drawing ───────────────────────────────────────────────
    def _draw(self):
        self.scr.erase()
        h, w = self.th, self.tw
        ba = curses.color_pair(CP_BORDER)

        # top border
        saddstr(self.scr, 0, 0, '\u2554' + '\u2550' * (w - 2) + '\u2557', ba)
        # side borders score area
        for y in range(1, SCORE_ROWS):
            saddstr(self.scr, y, 0, '\u2551', ba)
            saddstr(self.scr, y, w - 1, '\u2551', ba)
        # separator
        saddstr(self.scr, SCORE_ROWS, 0,
                '\u2560' + '\u2550' * (w - 2) + '\u2563', ba)

        # scores
        self._draw_scores()

        # field side borders
        for y in range(self.f_top, self.f_top + self.fh):
            saddstr(self.scr, y, 0, '\u2551', ba)
            saddstr(self.scr, y, w - 1, '\u2551', ba)

        # center line
        mid = w // 2
        for y in range(self.f_top, self.f_top + self.fh):
            if y % 2 == 0:
                saddstr(self.scr, y, mid, CH_CENTER,
                        curses.color_pair(CP_DIM) | curses.A_BOLD)

        # bottom separator + info
        bot = self.f_top + self.fh
        saddstr(self.scr, bot, 0,
                '\u2560' + '\u2550' * (w - 2) + '\u2563', ba)
        iy = bot + 1
        saddstr(self.scr, iy, 0, '\u2551', ba)
        saddstr(self.scr, iy, w - 1, '\u2551', ba)
        saddstr(self.scr, iy, 2, ' W/S:Move  P:Pause  M:Music  Q:Quit',
                curses.color_pair(CP_DIM) | curses.A_BOLD)
        ai_s = f'AI LVL:{self.ai.level} '
        saddstr(self.scr, iy, w - len(ai_s) - 2, ai_s,
                curses.color_pair(CP_DANGER))
        if iy + 1 < h:
            saddstr(self.scr, iy + 1, 0,
                    '\u255a' + '\u2550' * (w - 2) + '\u255d', ba)

        # trail
        for i, (tx, ty) in enumerate(self.trail):
            ry = self.f_top + int(ty)
            rx = self.f_left + int(tx)
            ch = CH_TRAIL[min(i, len(CH_TRAIL) - 1)]
            saddstr(self.scr, ry, rx, ch, curses.color_pair(CP_TRAIL))

        # ball
        saddstr(self.scr, self.f_top + int(self.by),
                self.f_left + int(self.bx), CH_BALL,
                curses.color_pair(CP_BALL) | curses.A_BOLD)

        # turbo indicator
        if self.bspd > BALL_SPD_INIT + 0.25:
            t = 'TURBO!'
            saddstr(self.scr, self.f_top, cx(w, len(t)), t,
                    curses.color_pair(CP_DANGER) | curses.A_BOLD)

        # paddles
        for py in range(PADDLE_H):
            saddstr(self.scr, self.f_top + int(self.py_p) + py,
                    self.f_left + self.px_p, CH_PAD,
                    curses.color_pair(CP_PAD_P) | curses.A_BOLD)
            saddstr(self.scr, self.f_top + int(self.py_ai) + py,
                    self.f_left + self.px_ai, CH_PAD,
                    curses.color_pair(CP_PAD_AI) | curses.A_BOLD)

        # effects
        self.fx.render(self.scr)

        # pause overlay
        if self.paused:
            bw = 14
            cy = h // 2
            cxv = cx(w, bw)
            saddstr(self.scr, cy - 1, cxv,
                    '\u2554' + '\u2550' * (bw - 2) + '\u2557',
                    curses.color_pair(CP_BRIGHT) | curses.A_BOLD)
            saddstr(self.scr, cy, cxv,
                    '\u2551' + ' PAUSED '.center(bw - 2) + '\u2551',
                    curses.color_pair(CP_BRIGHT) | curses.A_BOLD)
            saddstr(self.scr, cy + 1, cxv,
                    '\u255a' + '\u2550' * (bw - 2) + '\u255d',
                    curses.color_pair(CP_BRIGHT) | curses.A_BOLD)

    def _draw_scores(self):
        w = self.tw
        mid = w // 2
        saddstr(self.scr, 1, 2, 'PLAYER',
                curses.color_pair(CP_PAD_P) | curses.A_BOLD)
        saddstr(self.scr, 1, w - 5, 'CPU',
                curses.color_pair(CP_PAD_AI) | curses.A_BOLD)
        saddstr(self.scr, 1, mid, '\u2502',
                curses.color_pair(CP_DIM) | curses.A_BOLD)
        for y in range(2, SCORE_ROWS):
            saddstr(self.scr, y, mid, '\u2502',
                    curses.color_pair(CP_DIM) | curses.A_BOLD)

        # big digits
        ps = str(self.sc_p)
        pw = len(ps) * 4 - 1
        self._big_num(self.sc_p, 2, mid // 2 - pw // 2, CP_PAD_P)
        ai_s = str(self.sc_ai)
        aw = len(ai_s) * 4 - 1
        self._big_num(self.sc_ai, 2, mid + (w - mid) // 2 - aw // 2,
                      CP_PAD_AI)

    def _big_num(self, num, y, x, cp):
        for ch in str(num):
            g = DIGITS.get(ch, DIGITS['0'])
            for ri, row in enumerate(g):
                saddstr(self.scr, y + ri, x, row,
                        curses.color_pair(cp) | curses.A_BOLD)
            x += 4

    def _show_round(self):
        self._draw()
        h, w = self.th, self.tw
        t = f'ROUND {self.rnd}'
        bw = len(t) + 4
        cy = h // 2
        cxv = cx(w, bw)
        saddstr(self.scr, cy - 1, cxv,
                '\u2554' + '\u2550' * (bw - 2) + '\u2557',
                curses.color_pair(CP_BRIGHT) | curses.A_BOLD)
        saddstr(self.scr, cy, cxv,
                '\u2551 ' + t + ' \u2551',
                curses.color_pair(CP_SCORE) | curses.A_BOLD)
        saddstr(self.scr, cy + 1, cxv,
                '\u255a' + '\u2550' * (bw - 2) + '\u255d',
                curses.color_pair(CP_BRIGHT) | curses.A_BOLD)
        self.scr.refresh()
        time.sleep(1.0)

    # ── game loop ─────────────────────────────────────────────
    def _loop(self):
        self.scr.timeout(0)
        while self.sc_p < WINNING_SCORE and self.sc_ai < WINNING_SCORE:
            t0 = time.monotonic()

            # drain input
            while True:
                k = self.scr.getch()
                if k == -1:
                    break
                self._key(k)

            if not self.paused:
                self._step_ai()
                self._step_ball()
                self.fx.update()

            self._draw()
            self.scr.refresh()

            dt = time.monotonic() - t0
            rem = (1.0 / FPS) - dt
            if rem > 0.001:
                time.sleep(rem)

        return 'player' if self.sc_p >= WINNING_SCORE else 'ai'

    def _key(self, k):
        if k in (ord('q'), ord('Q')):
            raise KeyboardInterrupt
        if k in (ord('p'), ord('P')):
            self.paused = not self.paused
        elif k in (ord('m'), ord('M')):
            self.snd.toggle()
        elif not self.paused:
            if k in (ord('w'), ord('W'), curses.KEY_UP):
                self.py_p -= PLAYER_SPD
            elif k in (ord('s'), ord('S'), curses.KEY_DOWN):
                self.py_p += PLAYER_SPD
            self.py_p = max(0, min(self.fh - PADDLE_H, self.py_p))

    # ── win / lose screens ────────────────────────────────────
    def _win_screen(self):
        self.snd.win()
        self.scr.timeout(50)
        h, w = self.th, self.tw
        trophy = [
            "    ___________    ",
            "   '._==_==_=_.'   ",
            "   .-\\:      /-.  ",
            "  | (|:.     |) | ",
            "   '-|:.     |-'  ",
            "     \\::.    /    ",
            "      '::. .'     ",
            "        ) (       ",
            "      _.' '._     ",
            "     '-------'    ",
        ]
        title_rows = text_art('YOU WIN!')
        t0 = time.monotonic()
        fx = FX()
        while time.monotonic() - t0 < 6.0:
            self.scr.erase()
            t = time.monotonic() - t0
            # title
            cp = CP_SCORE if int(t * 4) % 2 else CP_BRIGHT
            ty = 2
            for row in title_rows:
                saddstr(self.scr, ty, cx(w, len(row)), row,
                        curses.color_pair(cp) | curses.A_BOLD)
                ty += 1
            # trophy
            ty += 1
            for line in trophy:
                saddstr(self.scr, ty, cx(w, len(line)), line,
                        curses.color_pair(CP_SCORE) | curses.A_BOLD)
                ty += 1
            # score
            sc = f'FINAL SCORE: {self.sc_p} - {self.sc_ai}'
            saddstr(self.scr, ty + 1, cx(w, len(sc)), sc,
                    curses.color_pair(CP_BRIGHT))
            al = f'AI LEVEL REACHED: {self.ai.level}'
            saddstr(self.scr, ty + 2, cx(w, len(al)), al,
                    curses.color_pair(CP_INFO))
            # fireworks
            if random.random() < 0.18:
                fxp = random.randint(5, w - 5)
                fyp = random.randint(2, h // 2)
                for _ in range(14):
                    ang = random.uniform(0, 2 * math.pi)
                    spd = random.uniform(0.3, 1.8)
                    fx.parts.append(Particle(
                        fxp, fyp, math.cos(ang) * spd,
                        math.sin(ang) * spd,
                        random.choice(PARTICLE_CH),
                        random.uniform(0.6, 1.8),
                        random.choice([CP_SCORE, CP_DANGER,
                                       CP_BRIGHT, CP_INFO]),
                    ))
            fx.update()
            fx.render(self.scr)
            self.scr.refresh()
            if self.scr.getch() != -1 and t > 1.5:
                break

    def _lose_screen(self):
        self.snd.lose()
        self.scr.timeout(50)
        h, w = self.th, self.tw
        go_rows = text_art('GAME')
        ov_rows = text_art('OVER')
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            self.scr.erase()
            t = time.monotonic() - t0
            gy = max(2, (h - 14) // 2)
            cp = CP_DANGER if int(t * 6) % 3 != 0 else CP_BRIGHT
            # GAME
            for i, row in enumerate(go_rows):
                off = random.randint(-2, 2) if random.random() < 0.12 else 0
                saddstr(self.scr, gy + i, cx(w, len(row)) + off, row,
                        curses.color_pair(cp) | curses.A_BOLD)
            # OVER
            for i, row in enumerate(ov_rows):
                off = random.randint(-2, 2) if random.random() < 0.12 else 0
                saddstr(self.scr, gy + 6 + i, cx(w, len(row)) + off, row,
                        curses.color_pair(cp) | curses.A_BOLD)
            # score
            sc = f'FINAL SCORE: {self.sc_p} - {self.sc_ai}'
            saddstr(self.scr, gy + 13, cx(w, len(sc)), sc,
                    curses.color_pair(CP_BRIGHT))
            # static noise
            if random.random() < 0.3:
                ny = random.randint(0, h - 1)
                ns = ''.join(random.choice('\u2591\u2592\u2593\u2588 ')
                             for _ in range(w))
                saddstr(self.scr, ny, 0, ns,
                        curses.color_pair(CP_DIM) | curses.A_BOLD)
            self.scr.refresh()
            if self.scr.getch() != -1 and t > 1.5:
                break

    # ── highscore screen ──────────────────────────────────────
    def _hs_screen(self):
        self.scr.timeout(120)
        h, w = self.th, self.tw
        final = self.sc_p * 100 + self.ai.level * 10
        hi = -1
        if self.hs.qualifies(final):
            self.scr.erase()
            m = '** NEW HIGH SCORE! **'
            saddstr(self.scr, 3, cx(w, len(m)), m,
                    curses.color_pair(CP_SCORE) | curses.A_BOLD)
            sv = f'SCORE: {final}'
            saddstr(self.scr, 5, cx(w, len(sv)), sv,
                    curses.color_pair(CP_BRIGHT))
            self.scr.refresh()
            initials = self.hs.enter_initials(self.scr)
            hi = self.hs.add(initials, final, self.ai.level)

        t0 = time.monotonic()
        while True:
            self.scr.erase()
            self.hs.render(self.scr, hi)
            p = 'PRESS ANY KEY'
            saddstr(self.scr, h - 3, cx(w, len(p)), p,
                    curses.color_pair(CP_DIM) | curses.A_BOLD)
            self.scr.refresh()
            if self.scr.getch() != -1 and time.monotonic() - t0 > 0.8:
                break

    # ── play again prompt ─────────────────────────────────────
    def _again(self):
        self.scr.timeout(120)
        h, w = self.th, self.tw
        while True:
            t = 'PLAY AGAIN? (Y/N)'
            y = h - 2
            if int(time.time() * 2) % 2:
                saddstr(self.scr, y, cx(w, len(t)), t,
                        curses.color_pair(CP_SCORE) | curses.A_BOLD)
            else:
                saddstr(self.scr, y, cx(w, len(t)), ' ' * len(t))
            self.scr.refresh()
            k = self.scr.getch()
            if k in (ord('y'), ord('Y')):
                return True
            if k in (ord('n'), ord('N'), ord('q'), ord('Q'), 27):
                return False

    # ── main flow ─────────────────────────────────────────────
    def run(self):
        try:
            if not self._intro():
                return
            while True:
                self._init_field()
                self.snd.start_music()
                self._show_round()
                winner = self._loop()
                self.snd.stop_music()
                if winner == 'player':
                    self._win_screen()
                else:
                    self._lose_screen()
                self._hs_screen()
                if not self._again():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.snd.cleanup()


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main(stdscr):
    RetroPong(stdscr).run()


if __name__ == '__main__':
    os.environ.setdefault('ESCDELAY', '25')
    curses.wrapper(main)
