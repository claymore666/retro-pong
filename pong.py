#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════╗
║         R E T R O   P O N G               ║
║   80s Arcade Style  ·  ML Opponent        ║
╚═══════════════════════════════════════════╝

A terminal-based Pong game rendered with curses and Unicode box-drawing
characters.  The AI opponent uses a hybrid approach: a rule-based
trajectory heuristic provides a competent baseline (~84 % hit rate at
level 1), while a neural network trained via Deep Q-Learning gradually
takes over as the AI levels up with each successful hit.

Architecture overview
─────────────────────
  Sound       – Procedural chiptune audio (square/triangle/noise waves)
                generated with NumPy and streamed to ``aplay`` via stdin.
  NN          – Minimal 2-layer MLP (forward + backprop) used by the AI.
  AI          – Deep Q-Learning agent with experience replay and a
                level-dependent heuristic that models human-like skill
                progression (beginner → expert).
  FX          – Particle system for explosions, screen-shake, flash,
                and timed popup messages.
  HiScores    – Persistent top-10 arcade highscore table stored as JSON
                in the user's home directory.
  RetroPong   – Main game class: intro sequence, game loop with 30 FPS
                fixed-timestep, physics, rendering, win/lose animations,
                and highscore entry.

Dependencies: Python 3.8+, NumPy.  Optional: ``aplay`` (ALSA) for sound.
"""

# ═══════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════

import curses          # Terminal UI: windows, colours, keyboard input
import locale          # Ensure UTF-8 so Unicode chars render correctly
import time            # Frame pacing via time.monotonic / time.sleep
import json            # Persist highscores as JSON
import math            # Trigonometry for ball angles, particle directions
import random          # Random angles, noise, AI exploration
import subprocess      # Spawn ``aplay`` for sound playback
import threading       # Background threads for non-blocking audio I/O
import io              # In-memory WAV byte buffer
import sys
import os
import wave as wave_mod          # Encode PCM samples as WAV
from collections import deque    # Fixed-size replay buffer for DQN
from datetime import datetime    # Timestamp highscore entries
from pathlib import Path         # Cross-platform path to scores file

# NumPy is the only external dependency.  It is used for:
#   1. Waveform generation (sound engine)
#   2. Neural-network forward pass and backpropagation (AI)
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

# Set locale so that the terminal knows we're using UTF-8 encoding.
# Without this, curses may refuse to render Unicode characters like
# ● █ ╔ ═ etc.
locale.setlocale(locale.LC_ALL, '')


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
#
#  All tunable game parameters are collected here so they can be
#  tweaked without hunting through the code.
# ═══════════════════════════════════════════════════════════════

FPS = 30                        # Target frames per second
WINNING_SCORE = 11              # First to this many points wins
PADDLE_H = 4                    # Paddle height in terminal rows
BALL_SPD_INIT = 0.55            # Ball speed at start of each rally (cells/frame)
BALL_SPD_MULT = 1.05            # Multiplicative speed increase per rally hit (+5 %)
BALL_SPD_MAX = 2.75             # Hard cap = 5× starting speed
PLAYER_SPD = 1.2                # Player paddle movement per key-press (cells)
AI_SPD_BASE = 0.85              # AI paddle base speed (unused directly; see AI.speed)
AI_SPD_INC = 0.01               # AI speed increment per level (unused; see AI.speed)
MIN_W, MIN_H = 60, 24           # Minimum terminal dimensions
SCORE_ROWS = 7                  # Rows reserved for the score display area

# ── Unicode characters used for rendering ─────────────────────
CH_BALL = '\u25cf'               # ●  Filled circle for the ball
CH_PAD = '\u2588'                # █  Full block for paddles
CH_TRAIL = ['\u25c9', '\u25cb', '\u00b7']  # ◉ ○ · Fading ball trail
CH_CENTER = '\u250a'             # ┊  Dashed vertical for center court line
PARTICLE_CH = ['*', '+', '\u00b7', '.', 'o']  # Characters for particle effects

# ── Curses colour-pair IDs ────────────────────────────────────
# Each ID maps to a (foreground, background) colour pair initialised
# in RetroPong._init_curses().  Using named constants avoids magic
# numbers scattered through the rendering code.
CP_NORM = 1;  CP_BRIGHT = 2;  CP_SCORE = 3;  CP_DANGER = 4
CP_INFO = 5;  CP_DIM = 6;     CP_PAD_P = 7;  CP_PAD_AI = 8
CP_BORDER = 9; CP_BALL = 10;  CP_TITLE = 11; CP_TRAIL = 12


# ═══════════════════════════════════════════════════════════════
#  FONTS
#
#  Two bitmap fonts rendered from █ (full-block) characters:
#    DIGITS – 3 columns × 5 rows, used for the LED-style scoreboard.
#    FONT   – 5 columns × 5 rows, used for large title text
#             ("RETRO PONG", "YOU WIN!", "GAME OVER").
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
    ' ': ['  ', '  ', '  ', '  ', '  '],  # Narrow space between words
}


def text_art(text):
    """Convert *text* into a list of 5 strings forming large block letters.

    Each character is looked up in FONT (5 wide × 5 tall).  Characters
    are separated by a single-column gap.  Unknown characters render as
    spaces.

    Returns:
        List of 5 strings, one per pixel row.

    Example::

        >>> text_art('HI')
        ['█   █ █████', '█   █   █  ', '█████   █  ',
         '█   █   █  ', '█   █ █████']
    """
    rows = [''] * 5
    for ch in text.upper():
        g = FONT.get(ch, FONT[' '])
        for i in range(5):
            if rows[i]:
                rows[i] += ' '       # 1-column gap between letters
            rows[i] += g[i]
    return rows


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def saddstr(win, y, x, s, attr=0):
    """Safe ``addstr`` that clips text to the curses window boundaries.

    Standard ``curses.addstr`` raises an error when writing past the
    bottom-right corner of the window.  This wrapper silently clips
    the string so we never need to worry about off-screen writes
    crashing the game.

    Args:
        win:  curses window object.
        y, x: row and column (0-based).
        s:    string to draw.
        attr: curses attribute flags (colour pair, bold, etc.).
    """
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        # Clip string that starts before the left edge
        if x < 0:
            s = s[-x:]
            x = 0
        # Clip string that extends past the right edge
        s = s[:w - x]
        if s:
            win.addstr(y, x, s, attr)
    except curses.error:
        # curses raises an error when writing to the very last cell
        # of the window (bottom-right).  The character *is* written,
        # so we can safely ignore the exception.
        pass


def cx(total, text_w):
    """Return the x-offset to horizontally centre *text_w* inside *total*."""
    return max(0, (total - text_w) // 2)


def predict_y(by, bdy, steps, fh):
    """Predict the ball's Y position after *steps* frames of movement.

    The ball bounces off the top (y=0) and bottom (y=fh-1) walls.
    Instead of simulating step-by-step, we use modular arithmetic:
    the ball's vertical motion is periodic with period 2*(fh-1).

    The "folding" trick:
      1. Compute raw_y = by + bdy * steps  (position without walls).
      2. Take raw_y modulo the period to wrap it into one cycle.
      3. If the wrapped value exceeds fh-1, the ball is on the
         "return leg" of the bounce — mirror it back.

    Args:
        by:    Current ball Y position.
        bdy:   Ball Y velocity (cells per frame).
        steps: Number of frames into the future.
        fh:    Field height (valid Y range is 0 .. fh-1).

    Returns:
        Predicted Y as a float, clamped to [0, fh-1].
    """
    if fh <= 1 or steps <= 0:
        return by
    raw = by + bdy * steps
    # One full bounce cycle: ball goes from 0 → fh-1 → 0 = 2*(fh-1) cells
    period = 2.0 * (fh - 1)
    if period <= 0:
        return by
    # Wrap into one period
    y = raw % period
    if y < 0:
        y += period
    # Mirror the return leg
    if y > fh - 1:
        y = period - y
    return max(0.0, min(float(fh - 1), y))


# ═══════════════════════════════════════════════════════════════
#  SOUND ENGINE  (chiptune via numpy + aplay)
#
#  All audio is generated procedurally using NumPy:
#    - Square waves  (_sq)  for the chiptune lead melody and SFX.
#    - Triangle waves (_tri) for the bass line.
#    - White noise   (_noise) for hi-hat percussion.
#
#  Playback uses the Linux ``aplay`` command (part of alsa-utils).
#  Sound effects are one-shot WAV buffers piped to short-lived aplay
#  processes.  Background music is raw PCM streamed in a loop to a
#  single long-lived aplay process via a daemon thread.
#
#  If aplay is not available (e.g. inside Docker without ALSA),
#  sound is silently disabled and the game works without audio.
# ═══════════════════════════════════════════════════════════════

class Sound:
    """Procedural chiptune sound engine.

    Attributes:
        RATE: Sample rate in Hz (22050 — adequate for chiptune).
        on:   Whether sound is currently enabled (toggled with M key).
    """
    RATE = 22050

    def __init__(self):
        self.on = True
        self._ok = self._probe()       # True if aplay is available
        self._music_proc = None         # subprocess for background music
        self._fx = []                   # list of one-shot SFX subprocesses

    # -- probing -------------------------------------------------------
    def _probe(self):
        """Check whether ``aplay`` is installed and functional.

        Returns True if ``aplay --version`` exits successfully.
        Any failure (FileNotFoundError, timeout, bad exit code)
        results in False and all sound methods become no-ops.
        """
        try:
            return subprocess.run(
                ['aplay', '--version'], capture_output=True, timeout=2
            ).returncode == 0
        except Exception:
            return False

    # -- waveform generators -------------------------------------------

    def _sq(self, freq, dur, vol=0.15):
        """Generate a square wave with attack/release envelope.

        A square wave alternates between +1 and -1 at the given
        frequency.  It produces the characteristic "buzzy" chiptune
        sound (think NES/Game Boy).

        The short attack (5 ms ramp-up) and release (10 ms ramp-down)
        envelope prevents audible clicks at note boundaries.

        Args:
            freq: Frequency in Hz (e.g. 440 for concert A).
            dur:  Duration in seconds.
            vol:  Volume as a fraction of full scale (0.0–1.0).

        Returns:
            NumPy int16 array of PCM samples.
        """
        t = np.linspace(0, dur, int(self.RATE * dur), False)
        # np.sign(sin(...)) gives +1 or -1 → square wave
        w = np.sign(np.sin(2 * np.pi * freq * t))
        n = len(w)
        # Build amplitude envelope to avoid clicks
        env = np.ones(n)
        a = min(int(0.005 * self.RATE), n)   # 5 ms attack
        r = min(int(0.01 * self.RATE), n)    # 10 ms release
        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if r > 0:
            env[-r:] = np.linspace(1, 0, r)
        return (w * env * vol * 32767).astype(np.int16)

    def _tri(self, freq, dur, vol=0.20):
        """Generate a triangle wave (softer tone for bass lines).

        A triangle wave ramps linearly between -1 and +1.  It has
        fewer harmonics than a square wave, producing a mellower,
        rounder sound suitable for bass.

        The formula ``2 * |2 * (t*f - floor(t*f + 0.5))| - 1`` maps
        the sawtooth phase into a triangle by mirroring it.

        Args:
            freq: Frequency in Hz.
            dur:  Duration in seconds.
            vol:  Volume (0.0–1.0).

        Returns:
            NumPy int16 array of PCM samples.
        """
        t = np.linspace(0, dur, int(self.RATE * dur), False)
        w = 2.0 * np.abs(2.0 * (t * freq - np.floor(t * freq + 0.5))) - 1.0
        return (w * vol * 32767).astype(np.int16)

    def _noise(self, dur, vol=0.06):
        """Generate white noise (for hi-hat / percussion sounds).

        Uniform random samples in [-1, +1] produce a hissing sound
        that works well for drum machine hi-hats.

        Args:
            dur: Duration in seconds.
            vol: Volume (0.0–1.0).

        Returns:
            NumPy int16 array of PCM samples.
        """
        n = int(self.RATE * dur)
        return (np.random.uniform(-1, 1, n) * vol * 32767).astype(np.int16)

    # -- playback ------------------------------------------------------

    def _play(self, samples):
        """Play a one-shot sound effect asynchronously.

        Encodes the samples as an in-memory WAV file and pipes it to
        a new ``aplay`` process.  A daemon thread feeds the data so
        the game loop is never blocked by audio I/O.

        Args:
            samples: NumPy int16 array of PCM audio data.
        """
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
        """Encode a NumPy int16 array as a WAV byte string.

        Uses Python's ``wave`` module to write a proper WAV header
        (mono, 16-bit, at self.RATE) into an in-memory BytesIO buffer.
        The returned bytes can be piped directly to ``aplay``.
        """
        buf = io.BytesIO()
        with wave_mod.open(buf, 'wb') as wf:
            wf.setnchannels(1)          # Mono
            wf.setsampwidth(2)          # 16-bit (2 bytes per sample)
            wf.setframerate(self.RATE)
            wf.writeframes(samples.tobytes())
        return buf.getvalue()

    # -- sound effects -------------------------------------------------

    def hit(self):
        """Short high-pitched blip when the ball hits a paddle."""
        self._play(self._sq(880, 0.05, 0.12))

    def wall(self):
        """Softer blip when the ball bounces off the top/bottom wall."""
        self._play(self._sq(440, 0.03, 0.08))

    def score(self):
        """Descending three-note buzz when a point is scored."""
        self._play(np.concatenate([
            self._sq(600, 0.08, 0.15),
            self._sq(400, 0.08, 0.15),
            self._sq(200, 0.16, 0.15),
        ]))

    def win(self):
        """Ascending triumphant jingle when the player wins the game."""
        self._play(np.concatenate([
            self._sq(f, 0.10, 0.18)
            for f in [523, 587, 659, 784, 880, 1047]  # C5→C6 scale
        ]))

    def lose(self):
        """Descending deflating jingle when the player loses."""
        self._play(np.concatenate([
            self._sq(f, 0.13, 0.18) for f in [400, 350, 300, 250, 200]
        ]))

    # -- background music ----------------------------------------------

    def start_music(self):
        """Start the background chiptune music loop.

        Generates the full track once, converts to raw PCM bytes,
        then spawns a single ``aplay`` process in raw-PCM mode.
        A daemon thread continuously writes the track buffer to
        aplay's stdin — when it finishes one pass, it starts again,
        creating a seamless loop.

        The raw-PCM approach (vs. WAV) avoids re-sending the WAV
        header on each loop iteration, which would cause a click.
        """
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
                    # Keep writing the track buffer; aplay's pipe buffer
                    # provides natural backpressure (write blocks when
                    # the buffer is full, matching playback speed).
                    while self._music_proc and self._music_proc.poll() is None:
                        self._music_proc.stdin.write(raw)
                        self._music_proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            threading.Thread(target=_loop, daemon=True).start()
        except Exception:
            self._music_proc = None

    def stop_music(self):
        """Kill the background music process."""
        if self._music_proc:
            try:
                self._music_proc.kill()
                self._music_proc.wait(timeout=1)
            except Exception:
                pass
            self._music_proc = None

    def toggle(self):
        """Toggle music on/off (bound to the M key)."""
        self.on = not self.on
        if self.on:
            self.start_music()
        else:
            self.stop_music()

    def cleanup(self):
        """Kill all audio subprocesses (called on game exit)."""
        self.stop_music()
        for p in self._fx:
            try:
                p.kill()
            except Exception:
                pass

    def _make_music(self):
        """Generate a complete chiptune soundtrack (~82 seconds).

        The track is structured like a real song rather than a short
        loop, with seven sections that share the same Cm→Bb→Ab→G
        chord progression but differ in melodic treatment:

            Intro   (4 bars)  – Bass + sparse hi-hat only; builds tension.
            Verse A (8 bars)  – Main theme: ascending Cm arpeggios.
            Verse A'(8 bars)  – Subtle variation: first phrase inverted.
            Chorus  (8 bars)  – Same chords, melody peaks an octave higher.
            Bridge  (8 bars)  – Half-time feel (notes held longer).
            Reprise (8 bars)  – Main theme returns unchanged.
            Outro   (4 bars)  – Melody thins out, bass walks to root.

        All audio is mixed from three channels:
            Lead  – Square wave melody (the main tune).
            Bass  – Triangle wave bass line (root notes).
            Hi-hat – White noise percussion pattern.

        Returns:
            NumPy int16 array containing the entire track as PCM audio.
        """
        bpm = 140
        n8 = 60.0 / bpm / 2          # Duration of one eighth note (~0.214 s)
        s8 = int(self.RATE * n8)      # Samples per eighth note

        # ── Note frequency table (Hz) ────────────────────────
        # Named as NoteOctave, e.g. C4 = middle C = 262 Hz.
        # fmt: off
        C2=65;   D2=73;   Eb2=78;  F2=87;   G2=98
        Ab2=104; Bb2=117; B2=123;  C3=131;  D3=147
        Eb3=156; F3=175;  G3=196;  Ab3=208; Bb3=233
        B3=247;  C4=262;  D4=294;  Eb4=311; F4=349
        G4=392;  Ab4=415; Bb4=466; B4=494;  C5=523
        D5=587;  Eb5=659; F5=698;  G5=784
        REST = 0   # Silence (no note played)
        # fmt: on

        def _render(mel, bass_notes, hat_pat, lead_vol=0.10,
                    bass_vol=0.18, hat_vol=0.04, note_len=0.85):
            """Render one section of music from melody + bass + hi-hat data.

            Args:
                mel:        List of frequencies, one per eighth note.
                bass_notes: List of frequencies, one per half note (4 eighths).
                hat_pat:    Binary pattern (1=hit, 0=rest) that repeats every
                            bar (8 eighths).
                lead_vol:   Lead channel volume (0.0–1.0).
                bass_vol:   Bass channel volume.
                hat_vol:    Hi-hat channel volume.
                note_len:   Fraction of the eighth note duration that the lead
                            note sustains (0.85 = slight staccato gap).

            Returns:
                NumPy int16 array of mixed PCM audio for this section.
            """
            n = len(mel)
            total = s8 * n
            lead = np.zeros(total, dtype=np.int16)
            bass = np.zeros(total, dtype=np.int16)
            hh   = np.zeros(total, dtype=np.int16)

            # Lead melody: one note per eighth note
            for i, f in enumerate(mel):
                off = i * s8
                if f > 0:
                    s = self._sq(f, n8 * note_len, lead_vol)
                    e = min(off + len(s), total)
                    lead[off:e] = s[:e - off]

            # Bass line: one note per half note (spans 4 eighths).
            # Duration is 3.9 eighths — leaves a tiny natural gap
            # before the next bass note for rhythmic separation.
            for i, f in enumerate(bass_notes):
                off = i * s8 * 4
                if f > 0:
                    s = self._tri(f, n8 * 3.9, bass_vol)
                    e = min(off + len(s), total)
                    bass[off:e] = s[:e - off]

            # Hi-hat: the pattern repeats every len(hat_pat) eighths.
            # Typically 8 entries = one bar.
            for i in range(n):
                pi = i % len(hat_pat)
                if hat_pat[pi]:
                    off = i * s8
                    s = self._noise(n8 * 0.25, hat_vol)
                    e = min(off + len(s), total)
                    hh[off:e] = s[:e - off]

            # Mix all three channels, clipping to int16 range
            return np.clip(
                lead.astype(np.int32) + bass.astype(np.int32)
                + hh.astype(np.int32), -32767, 32767
            ).astype(np.int16)

        # ── INTRO (4 bars = 32 eighths) ──────────────────────
        # No melody — just bass and sparse hi-hat to build tension
        # before the main theme kicks in.
        intro_mel  = [REST]*32
        intro_bass = [C2, C2, C2, C2, Ab2, Ab2, G2, G2]
        intro_hat  = [1,0,0,0, 1,0,0,0]
        intro = _render(intro_mel, intro_bass, intro_hat, hat_vol=0.05)

        # ── VERSE A (8 bars = 64 eighths) ────────────────────
        # The main theme: ascending arpeggios through Cm→Bb→Ab→G.
        # Each chord gets 2 bars (16 eighth notes).
        va_mel = [
            # Cm (2 bars): C4→Eb4→G4→C5 arpeggio, then back down
            C4, Eb4, G4, C5, G4, Eb4, C4, Eb4,
            C4, G4,  C5, G4, Eb4, C4, Eb4, G4,
            # Bb (2 bars)
            Bb3, D4, F4, Bb4, F4, D4, Bb3, D4,
            Bb3, F4, Bb4, F4, D4, Bb3, D4, F4,
            # Ab (2 bars)
            Ab3, C4, Eb4, Ab4, Eb4, C4, Ab3, C4,
            Ab3, Eb4, Ab4, Eb4, C4, Ab3, C4, Eb4,
            # G → Cm resolve (2 bars)
            G3, B3, D4, G4, D4, B3, G3, B3,
            G3, D4, G4, B4, G4, D4, C4, Eb4,
        ]
        # 16 bass entries = 16 half-notes = 8 bars (matches melody)
        va_bass = [C2, C2, C2, C3, Bb2, Bb2, Bb2, Bb2,
                   Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C2]
        va_hat  = [1,0,1,0, 1,0,1,0]    # Steady eighth-note hi-hat
        verse_a = _render(va_mel, va_bass, va_hat)

        # ── VERSE A' (8 bars) ────────────────────────────────
        # Subtle variation: first phrase of each chord starts from
        # the top of the arpeggio instead of the bottom, while the
        # second phrase stays the same.  Keeps it recognisable.
        va2_mel = [
            # Cm — starts high, descends (inversion of verse A)
            C5, G4, Eb4, C4, Eb4, G4, C5, G4,
            C4, G4,  C5, G4, Eb4, C4, Eb4, G4,
            # Bb — starts from top
            Bb4, F4, D4, Bb3, D4, F4, Bb4, F4,
            Bb3, F4, Bb4, F4, D4, Bb3, D4, F4,
            # Ab — identical to verse A (acts as an anchor)
            Ab3, C4, Eb4, Ab4, Eb4, C4, Ab3, C4,
            Ab3, Eb4, Ab4, Eb4, C4, Ab3, C4, Eb4,
            # G → Cm — slight embellishment at the very end
            G3, B3, D4, G4, D4, B3, G3, B3,
            G3, D4, G4, B4, G4, Eb4, C4, G4,
        ]
        va2_bass = [C2, C2, C2, C3, Bb2, Bb2, Bb2, Bb2,
                    Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C2]
        va2_hat  = [1,0,1,0, 1,0,1,1]   # Extra hat tick on beat 4-and
        verse_a2 = _render(va2_mel, va2_bass, va2_hat)

        # ── CHORUS (8 bars) ──────────────────────────────────
        # Same chord progression, but the melody reaches an octave
        # higher at the end of each 2-bar phrase.  Provides a subtle
        # energy lift without sounding like a different song.
        ch_mel = [
            # Cm — bar 2 peaks at Eb5
            C4, Eb4, G4, C5, G4, Eb4, C4, G4,
            C4, Eb4, G4, C5, Eb5, C5, G4, Eb4,
            # Bb — bar 2 peaks at D5
            Bb3, D4, F4, Bb4, F4, D4, Bb3, F4,
            Bb3, D4, F4, Bb4, D5, Bb4, F4, D4,
            # Ab — bar 2 peaks at C5
            Ab3, C4, Eb4, Ab4, Eb4, C4, Ab3, Eb4,
            Ab3, C4, Eb4, Ab4, C5, Ab4, Eb4, C4,
            # G → Cm — builds to the highest note (G5) then resolves
            G3, B3, D4, G4, D4, B3, G4, B4,
            G4, B4, D5, G5, D5, B4, C5, Eb5,
        ]
        ch_bass = [C2, C2, C2, C3, Bb2, Bb2, Bb2, Bb2,
                   Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C3]
        ch_hat  = [1,0,1,1, 1,0,1,1]    # Slightly busier hi-hat
        chorus = _render(ch_mel, ch_bass, ch_hat,
                         lead_vol=0.12, hat_vol=0.05)

        # ── BRIDGE (8 bars) ──────────────────────────────────
        # Half-time feel: each note is repeated to last a quarter
        # note (2 eighths) instead of an eighth.  Uses the same
        # chords but gives breathing room after the chorus energy.
        br_mel = [
            # Cm — long ascending tones
            C4, C4, Eb4, Eb4, G4, G4, C5, C5,
            G4, G4, Eb4, Eb4, C4, C4, C4, C4,
            # Bb — long tones
            Bb3, Bb3, D4, D4, F4, F4, Bb4, Bb4,
            F4, F4, D4, D4, Bb3, Bb3, Bb3, Bb3,
            # Ab — long tones
            Ab3, Ab3, C4, C4, Eb4, Eb4, Ab4, Ab4,
            Eb4, Eb4, C4, C4, Ab3, Ab3, Ab3, Ab3,
            # G → Cm — builds back into the main theme
            G3, G3, B3, B3, D4, D4, G4, G4,
            G4, G4, B4, B4, D4, D4, C4, Eb4,
        ]
        br_bass = [C2, C2, C2, C2, Bb2, Bb2, Bb2, Bb2,
                   Ab2, Ab2, Ab2, Ab2, G2, G2, G2, C2]
        br_hat  = [1,0,0,0, 1,0,0,0]    # Sparse — matches half-time feel
        bridge = _render(br_mel, br_bass, br_hat,
                         lead_vol=0.09, bass_vol=0.20, note_len=0.95)

        # ── VERSE A reprise (8 bars) ─────────────────────────
        # The main theme returns unchanged after the bridge, giving
        # the listener a satisfying "homecoming" feeling.
        reprise = _render(va_mel, va_bass, va_hat)

        # ── OUTRO (4 bars = 32 eighths) ──────────────────────
        # Melody progressively drops out (more REST gaps).
        # Bass walks chromatically back down to C, signalling
        # the end of the loop before it seamlessly restarts.
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

        # ── Assemble all sections into the final track ────────
        return np.concatenate([
            intro, verse_a, verse_a2, chorus,
            bridge, reprise, outro,
        ])


# ═══════════════════════════════════════════════════════════════
#  NEURAL NETWORK  (2-layer MLP)
#
#  A minimal feedforward neural network implemented from scratch
#  with NumPy.  Used by the AI opponent for Q-value prediction.
#
#  Architecture: Input(6) → Dense(128, ReLU) → Dense(3, linear)
#
#  Training uses vanilla stochastic gradient descent on MSE loss
#  with manual backpropagation.  No autograd framework needed.
# ═══════════════════════════════════════════════════════════════

class NN:
    """Two-layer multi-layer perceptron (MLP) for Q-learning.

    The network maps a 6-dimensional game state to 3 Q-values,
    one per possible action (move up, stay, move down).

    Weight initialisation uses He/Kaiming init (variance = 2/fan_in)
    which works well with ReLU activations.

    Args:
        ni: Number of input features (default 6).
        nh: Number of hidden units (default 64).
        no: Number of outputs / actions (default 3).
        lr: Learning rate for SGD (default 0.002).
    """

    def __init__(self, ni=6, nh=64, no=3, lr=0.002):
        # He initialisation: scale by sqrt(2/fan_in)
        self.W1 = np.random.randn(ni, nh) * np.sqrt(2.0 / ni)
        self.b1 = np.zeros(nh)
        self.W2 = np.random.randn(nh, no) * np.sqrt(2.0 / nh)
        self.b2 = np.zeros(no)
        self.lr = lr

    def forward(self, X):
        """Forward pass: X → hidden (ReLU) → output (linear).

        Intermediate activations are cached in self._X, self._Z1,
        self._A1 for use during backpropagation in train().

        Args:
            X: Input array, shape (batch, ni) or (ni,).

        Returns:
            Output Q-values, shape (batch, no).
        """
        X = np.atleast_2d(X)
        self._X = X
        self._Z1 = X @ self.W1 + self.b1          # Pre-activation
        self._A1 = np.maximum(0, self._Z1)         # ReLU activation
        return self._A1 @ self.W2 + self.b2        # Linear output

    def train(self, X, T):
        """One gradient-descent step on MSE loss.

        Performs forward pass, computes loss, then backpropagates
        gradients through both layers and updates weights in-place.

        Backprop derivation (MSE loss = mean((P - T)^2)):
          dL/dZ2 = (P - T) / batch_size
          dL/dW2 = A1^T · dZ2
          dL/dA1 = dZ2 · W2^T
          dL/dZ1 = dA1 * (Z1 > 0)        ← ReLU derivative
          dL/dW1 = X^T · dZ1

        Args:
            X: Input batch, shape (batch, ni).
            T: Target Q-values, shape (batch, no).

        Returns:
            Scalar MSE loss (float).
        """
        X = np.atleast_2d(X)
        T = np.atleast_2d(T)
        P = self.forward(X)
        d = P - T                       # Prediction error
        bs = X.shape[0]

        # ── Backpropagation ──
        dZ2 = d / bs                    # Gradient of loss w.r.t. output
        dW2 = self._A1.T @ dZ2
        db2 = dZ2.sum(0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self._Z1 > 0)     # ReLU derivative: 1 if Z1>0, else 0
        dW1 = self._X.T @ dZ1
        db1 = dZ1.sum(0)

        # ── SGD weight update ──
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return float(np.mean(d ** 2))


# ═══════════════════════════════════════════════════════════════
#  AI OPPONENT  (DQN with pre-training)
#
#  The AI uses a HYBRID approach:
#
#  1. Rule-based heuristic (_rule) — provides a reliable baseline
#     whose skill varies by level:
#       Lvl 1-2: tracks ball's current Y only (beginner)
#       Lvl 3-4: linear extrapolation, no bounce awareness
#       Lvl 5-7: full bounce prediction, moderate noise
#       Lvl 8+:  near-perfect trajectory prediction
#
#  2. Neural network (DQN) — trained via experience replay during
#     gameplay.  Its influence (nn_mix) increases with each hit.
#
#  The blend ensures the AI is immediately competent (~84 % at
#  level 1) while genuinely improving through machine learning.
# ═══════════════════════════════════════════════════════════════

class AI:
    """AI opponent with human-like skill progression.

    The AI levels up every 3 successful hits.  Each level increases:
      - Movement speed
      - Prediction accuracy (less noise, better trajectory model)
      - Reaction speed (responds to ball earlier)
      - Neural network influence (nn_mix)

    Attributes:
        ACTIONS: Maps action index to paddle movement: [-1, 0, +1].
        hits:    Total successful paddle hits (never resets).
        level:   Current difficulty level (1 + hits // 3).
        nn_mix:  Probability of using NN instead of rule (0.0–0.85).
    """
    ACTIONS = [-1, 0, 1]   # up / stay / down

    def __init__(self, fw, fh):
        """Initialise the AI for a field of size fw × fh.

        Args:
            fw: Field width in cells.
            fh: Field height in cells.
        """
        self.fw, self.fh = fw, fh
        self.nn = NN(6, 128, 3)         # Larger hidden layer for Q-learning
        self.buf = deque(maxlen=5000)    # Experience replay buffer
        self.eps = 0.04                  # Exploration rate (epsilon-greedy)
        self.gamma = 0.9                 # Discount factor for future rewards
        self.hits = 0                    # Lifetime hit count
        self.level = 1                   # Difficulty level
        self.last_s = None               # Last observed state (for recording)
        self.last_a = None               # Last chosen action
        self.nn_mix = 0.0                # NN influence (starts at 0 = pure rule)
        self._pretrain()

    def _state(self, bx, by, bdx, bdy, acy):
        """Encode the game state as a normalised 6D feature vector.

        Features (all normalised to roughly [0, 1] or [-1, 1]):
          [0] ball_x / field_width      — horizontal position
          [1] ball_y / field_height     — vertical position
          [2] ball_dx / max_speed       — horizontal velocity
          [3] ball_dy / max_speed       — vertical velocity
          [4] ai_paddle_center / height — paddle position
          [5] distance_to_paddle / width — how far ball is from AI
        """
        return np.array([
            bx / max(self.fw, 1),
            by / max(self.fh, 1),
            bdx / BALL_SPD_MAX,
            bdy / BALL_SPD_MAX,
            acy / max(self.fh, 1),
            max(0, self.fw - bx) / max(self.fw, 1),
        ])

    # ── Level-aware heuristic ─────────────────────────────────

    def _rule(self, bx, by, bdx, bdy, acy):
        """Choose an action using a level-dependent heuristic.

        This function models how a human player improves:
          - Beginners just follow the ball with their eyes (current Y).
          - Intermediates learn to predict straight-line trajectories.
          - Experts can predict bounces off walls.

        Several "human" imperfections scale with level:
          - Attention threshold: ignores distant ball at low levels.
          - Hesitation: randomly freezes (fails to act).
          - Positional noise: Gaussian jitter on the predicted target.
          - Tracking bias: blends prediction toward current ball Y.
          - Dead zone: won't move unless offset exceeds threshold.

        Args:
            bx, by: Ball position.
            bdx, bdy: Ball velocity (cells/frame, already × speed).
            acy: AI paddle centre Y.

        Returns:
            Action index: 0 (up), 1 (stay), or 2 (down).
        """
        lvl = self.level

        # Ball heading away from AI — low-level AI just stands still,
        # higher-level AI drifts toward centre to prepare.
        if bdx <= 0.01:
            if lvl < 5:
                return 1
            ty = self.fh / 2.0
            diff = ty - acy
            return 0 if diff < -2 else (2 if diff > 2 else 1)

        # ── Attention threshold ──
        # Low-level AI ignores the ball until it crosses past a certain
        # x position.  This models the beginner habit of "waiting to
        # see what happens" before reacting.
        #   Lvl 1 → ball must pass 55 % of field width
        #   Lvl 8+ → always attentive (threshold = 0)
        react_x = self.fw * max(0.0, 0.65 - lvl * 0.08)
        if bx < react_x:
            return 1

        # ── Hesitation ──
        # Sometimes the AI just freezes instead of moving.
        #   Lvl 1 → 10 % freeze chance;  Lvl 8+ → 0 %
        hesitate = max(0.0, 0.12 - lvl * 0.015)
        if random.random() < hesitate:
            return 1

        # How many frames until ball reaches AI's x position
        steps = max(1, (self.fw - 3 - bx) / bdx)

        # ── Prediction quality ──
        if lvl <= 2:
            # Beginner: just track ball's current Y (no prediction)
            ty = by
        elif lvl <= 4:
            # Novice: linear extrapolation (ty = by + bdy*steps).
            # This is accurate for straight shots but completely wrong
            # when the ball bounces off a wall — the AI overshoots.
            ty = by + bdy * steps
            ty = max(0.0, min(float(self.fh - 1), ty))
        elif lvl <= 7:
            # Intermediate: full bounce-aware prediction
            ty = predict_y(by, bdy, steps, self.fh)
        else:
            # Expert: same prediction, but with much less noise below
            ty = predict_y(by, bdy, steps, self.fh)

        # ── Positional noise ──
        # Gaussian jitter added to the target position.  Even if the
        # prediction is perfect, noise makes the AI miss occasionally.
        #   Lvl 1 → sigma ≈ 3.5 cells  (often off by several cells)
        #   Lvl 10 → sigma ≈ 0.3 cells (nearly perfect)
        noise = max(0.3, 4.0 - lvl * 0.4)
        ty += random.gauss(0, noise)

        # ── Tracking bias ──
        # At low levels, the AI partially ignores its prediction and
        # just moves toward where the ball is *right now*.  This models
        # the beginner tendency to "follow the ball" rather than
        # anticipate where it will be.
        #   Lvl 1 → 50 % current Y + 50 % predicted
        #   Lvl 8+ → 100 % predicted
        track = max(0.0, 0.6 - lvl * 0.08)
        ty = ty * (1.0 - track) + by * track

        ty = max(0.0, min(float(self.fh - 1), ty))

        # ── Dead zone ──
        # The AI won't move unless the target is far enough from its
        # current position.  Prevents twitchy oscillation and makes
        # low-level AI look "lazy".
        #   Lvl 1 → 2.2 cells;  Lvl 10 → 0.6 cells
        deadzone = max(0.6, 2.5 - lvl * 0.2)
        diff = ty - acy
        if diff < -deadzone:
            return 0    # Move up
        if diff > deadzone:
            return 2    # Move down
        return 1        # Stay

    # ── Pre-training ──────────────────────────────────────────

    def _pretrain(self):
        """Pre-train the neural network on rule-based demonstrations.

        Generates 5000 random game states, labels each with the action
        the rule-based heuristic would choose, then trains the NN for
        100 epochs in mini-batches of 64.

        This gives the NN a reasonable starting policy so that when
        nn_mix increases (NN gets used more), it doesn't start from
        scratch.  The NN then refines this policy through live DQN
        training during gameplay.
        """
        S, T = [], []
        for _ in range(5000):
            # Random ball state heading toward AI
            bx = random.uniform(0, self.fw)
            by = random.uniform(1, self.fh - 1)
            spd = random.uniform(BALL_SPD_INIT, BALL_SPD_MAX)
            ang = random.uniform(-math.pi / 3, math.pi / 3)
            bdx = abs(spd * math.cos(ang))
            bdy = spd * math.sin(ang)
            acy = random.uniform(PADDLE_H / 2, self.fh - PADDLE_H / 2)

            # Get the rule's action as the label
            opt = self._rule(bx, by, bdx, bdy, acy)
            s = self._state(bx, by, bdx, bdy, acy)

            # Q-value targets: +1.0 for optimal action, -1.0 for others
            t = np.array([-1.0, -1.0, -1.0])
            t[opt] = 1.0
            S.append(s)
            T.append(t)

        S, T = np.array(S), np.array(T)
        # Train for 100 epochs with shuffled mini-batches
        for _ in range(100):
            idx = np.random.permutation(len(S))
            for i in range(0, len(S), 64):
                b = idx[i:i + 64]
                self.nn.train(S[b], T[b])

    # ── Action selection ──────────────────────────────────────

    def act(self, bx, by, bdx, bdy, acy):
        """Choose an action using the hybrid epsilon-greedy policy.

        Decision hierarchy:
          1. With probability *eps*: random action (exploration).
          2. With probability *nn_mix*: use neural network's argmax.
          3. Otherwise: use the rule-based heuristic.

        This ensures the AI always has a competent baseline (the rule)
        while the NN gradually earns more influence as it trains.

        Args:
            bx, by: Ball position.
            bdx, bdy: Ball velocity (actual, i.e. direction × speed).
            acy: AI paddle centre Y.

        Returns:
            Action index: 0 (up), 1 (stay), or 2 (down).
        """
        s = self._state(bx, by, bdx, bdy, acy)
        if random.random() < self.eps:
            a = random.randrange(3)              # Random exploration
        elif random.random() < self.nn_mix:
            a = int(np.argmax(self.nn.forward(s)[0]))  # NN policy
        else:
            a = self._rule(bx, by, bdx, bdy, acy)     # Rule fallback
        self.last_s, self.last_a = s, a
        return a

    def record(self, s, a, r, ns, done):
        """Store a (state, action, reward, next_state, done) transition
        in the experience replay buffer for later training."""
        self.buf.append((s, a, r, ns, done))

    def on_hit(self):
        """Called when the AI successfully hits the ball.

        Updates:
          - hits counter and level (every 3 hits = 1 level)
          - Decays exploration rate (eps) so AI explores less over time
          - Increases nn_mix so the NN gets used more often
          - Records a positive terminal reward (+1.0)
          - Triggers a mini-batch training step
        """
        self.hits += 1
        self.level = 1 + self.hits // 3
        self.eps = max(0.01, self.eps * 0.95)
        self.nn_mix = min(0.85, self.nn_mix + 0.03)  # +3 % NN influence
        if self.last_s is not None:
            # Terminal reward: ball was hit → +1.0
            self.record(self.last_s, self.last_a, 1.0, self.last_s, True)
        self._learn()

    def on_miss(self):
        """Called when the AI fails to hit the ball (opponent scores).

        Records a negative terminal reward (-1.0) and trains.
        """
        if self.last_s is not None:
            self.record(self.last_s, self.last_a, -1.0, self.last_s, True)
        self._learn()

    def _learn(self):
        """Train the NN on a mini-batch from the replay buffer.

        Uses the standard DQN update rule:
          Q_target[a] = reward              (if terminal)
          Q_target[a] = reward + γ·max(Q(s'))  (otherwise)

        Only the Q-value for the action actually taken is updated;
        other action Q-values remain at their current prediction.
        """
        if len(self.buf) < 32:
            return
        batch = random.sample(list(self.buf), min(64, len(self.buf)))
        Sb = np.array([t[0] for t in batch])     # States
        Ab = [t[1] for t in batch]                # Actions
        Rb = [t[2] for t in batch]                # Rewards
        Nb = np.array([t[3] for t in batch])      # Next states
        Db = [t[4] for t in batch]                # Done flags

        cq = self.nn.forward(Sb).copy()           # Current Q-values
        nq = self.nn.forward(Nb)                   # Next-state Q-values

        # Build targets: only modify the Q-value for the taken action
        tgt = cq.copy()
        for i in range(len(batch)):
            if Db[i]:
                tgt[i, Ab[i]] = Rb[i]             # Terminal: just reward
            else:
                tgt[i, Ab[i]] = Rb[i] + self.gamma * np.max(nq[i])
        self.nn.train(Sb, tgt)

    @property
    def speed(self):
        """AI paddle movement speed, increasing with level.

        Lvl 1 → 0.65 cells/frame    (slow, easy to outrun)
        Lvl 5 → 0.85 cells/frame
        Lvl 10 → 1.10 cells/frame
        Capped at 1.2 to keep things fair.
        """
        return min(1.2, 0.60 + self.level * 0.05)


# ═══════════════════════════════════════════════════════════════
#  PARTICLES & EFFECTS
#
#  Simple particle system for visual juice: explosions when
#  points are scored, screen-shake, flash, and timed popup
#  messages (e.g. "AI EVOLVED!").
# ═══════════════════════════════════════════════════════════════

class Particle:
    """A single particle with position, velocity, character, lifetime, and colour.

    Uses __slots__ for memory efficiency (many particles can exist at once).
    """
    __slots__ = ('x', 'y', 'dx', 'dy', 'ch', 'life', 'mlife', 'cp')

    def __init__(self, x, y, dx, dy, ch='*', life=1.0, cp=CP_BRIGHT):
        self.x = float(x); self.y = float(y)
        self.dx = dx; self.dy = dy       # Velocity (cells/frame)
        self.ch = ch                     # Character to render
        self.life = life                 # Remaining lifetime (seconds)
        self.mlife = life                # Initial lifetime (for fade calc)
        self.cp = cp                     # Curses colour pair ID


class FX:
    """Visual effects manager: particles, screen-shake, flash, popups.

    All effects are updated and rendered each frame by the game loop.
    """

    def __init__(self):
        self.parts = []      # Active particles
        self.shake = 0       # Screen-shake frames remaining
        self.flash = 0       # Screen-flash frames remaining
        self.msgs = []       # Timed popup messages: (text, y, x, ttl, colour_pair)

    def explode(self, x, y, n=14):
        """Spawn *n* particles radiating outward from (x, y).

        Each particle gets a random direction, speed, character,
        lifetime, and colour.  Used when a point is scored.
        """
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
        """Show a timed text message at (y, x) for *ttl* frames."""
        self.msgs.append((text, y, x, ttl, cp))

    def update(self):
        """Advance all effects by one frame.

        Particles move, experience gravity (dy += 0.05), and age.
        Dead particles are removed.  Shake/flash counters decrement.
        Expired messages are removed.
        """
        alive = []
        for p in self.parts:
            p.x += p.dx; p.y += p.dy
            p.dy += 0.05              # Simple downward gravity
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
        """Draw all active particles and popup messages to the window.

        Particles fade from their character to '.' to ' ' as they age.
        """
        for p in self.parts:
            ry, rx = int(p.y), int(p.x)
            # Fade: full char > 50 % life, dot > 20 %, invisible below
            a = p.life / p.mlife
            ch = p.ch if a > 0.5 else ('.' if a > 0.2 else ' ')
            saddstr(win, ry, rx, ch, curses.color_pair(p.cp))
        for t, y, x, ttl, c in self.msgs:
            saddstr(win, y, x, t, curses.color_pair(c) | curses.A_BOLD)


# ═══════════════════════════════════════════════════════════════
#  HIGH-SCORE TABLE
#
#  Persistent top-10 arcade-style highscore table.
#  Scores are stored as JSON in ~/.retropong_scores.json.
#  Features classic 3-letter initial entry with arrow-key selection.
# ═══════════════════════════════════════════════════════════════

class HiScores:
    """Manages loading, saving, displaying, and entering highscores.

    Score formula: player_points × 100 + ai_level × 10.
    This rewards both winning decisively and pushing the AI to
    higher difficulty levels.
    """
    FILE = Path.home() / '.retropong_scores.json'
    MAX = 10    # Maximum entries in the table

    def __init__(self):
        self.data = self._load()

    def _load(self):
        """Load scores from disk, returning an empty list on any error."""
        try:
            return json.loads(self.FILE.read_text())[:self.MAX]
        except Exception:
            return []

    def _save(self):
        """Persist the current score list to disk as pretty JSON."""
        try:
            self.FILE.write_text(json.dumps(self.data, indent=2))
        except Exception:
            pass

    def qualifies(self, score):
        """Return True if *score* would make it into the top 10."""
        return len(self.data) < self.MAX or score > min(
            (e['score'] for e in self.data), default=0)

    def add(self, initials, score, ai_lvl):
        """Insert a new score entry, sort descending, trim to MAX.

        Returns the index of the newly inserted entry (for highlighting).
        """
        e = {'initials': initials, 'score': score,
             'ai_level': ai_lvl,
             'date': datetime.now().strftime('%Y-%m-%d')}
        self.data.append(e)
        self.data.sort(key=lambda x: -x['score'])
        self.data = self.data[:self.MAX]
        self._save()
        return self.data.index(e)

    def enter_initials(self, win):
        """Classic arcade 3-letter initial entry screen.

        The player cycles through characters with Up/Down arrows
        and advances to the next position with Right/Enter.
        The current position blinks.

        Args:
            win: curses window for input and display.

        Returns:
            A 3-character string like "AAA" or "BOB".
        """
        h, w = win.getmaxyx()
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        sel = [0, 0, 0]      # Index into letters for each position
        pos = 0               # Currently active position (0–2)
        win.timeout(120)
        while True:
            cy, cxv = h // 2, w // 2
            # Clear the entry area
            for row in range(cy - 6, cy + 8):
                saddstr(win, row, 0, ' ' * w)

            saddstr(win, cy - 4, cx(w, 19), 'ENTER YOUR INITIALS',
                    curses.color_pair(CP_SCORE) | curses.A_BOLD)

            # Draw the three letter positions
            for i in range(3):
                ch = letters[sel[i]]
                x = cxv - 5 + i * 4
                attr = curses.color_pair(CP_BRIGHT) | curses.A_BOLD
                # Blink the active position by toggling reverse video
                if i == pos and int(time.time() * 4) % 2:
                    attr |= curses.A_REVERSE
                # Show up/down arrows on the active position
                if i == pos:
                    saddstr(win, cy - 1, x, '\u25b2',       # ▲
                            curses.color_pair(CP_INFO))
                    saddstr(win, cy + 1, x, '\u25bc',       # ▼
                            curses.color_pair(CP_INFO))
                saddstr(win, cy, x, ch, attr)
                saddstr(win, cy + 2, x, '\u2500',           # ─ underline
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
                    break   # All 3 positions confirmed
            elif k == curses.KEY_LEFT and pos > 0:
                pos -= 1

        return ''.join(letters[i] for i in sel)

    def render(self, win, hl=-1):
        """Draw the highscore table centred on screen.

        Args:
            win: curses window.
            hl:  Index of a newly added entry to highlight (blinks).
        """
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
            # Blink the highlighted entry
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
#
#  RetroPong ties everything together: intro → game loop → end
#  screen → highscore → play again.  The game loop runs at a
#  fixed 30 FPS with non-blocking input, float-precision ball
#  physics, and frame-rate-independent rendering.
# ═══════════════════════════════════════════════════════════════

class RetroPong:
    """Main game controller.

    Owns all subsystems (sound, effects, highscores, AI) and
    manages the game lifecycle from intro to exit.
    """

    def __init__(self, scr):
        """Initialise the game with the given curses screen.

        Args:
            scr: The root curses window (from curses.wrapper).
        """
        self.scr = scr
        self.snd = Sound()       # Chiptune sound engine
        self.fx = FX()           # Particle/effects system
        self.hs = HiScores()     # Persistent highscore table
        self._init_curses()

    # ── Curses setup ──────────────────────────────────────────

    def _init_curses(self):
        """Configure curses: hide cursor, enable colours, define colour pairs.

        Each colour pair maps a foreground colour to the default terminal
        background (-1).  These pairs are referenced throughout the rendering
        code via the CP_* constants.
        """
        curses.curs_set(0)           # Hide the blinking cursor
        self.scr.nodelay(True)       # getch() returns -1 immediately if no key
        self.scr.keypad(True)        # Enable arrow-key escape sequences
        curses.start_color()
        curses.use_default_colors()  # -1 = terminal's default background
        pairs = [
            (CP_NORM,   curses.COLOR_GREEN,   -1),    # Normal text (CRT green)
            (CP_BRIGHT, curses.COLOR_WHITE,   -1),    # Bright/highlighted text
            (CP_SCORE,  curses.COLOR_YELLOW,  -1),    # Score flashes, titles
            (CP_DANGER, curses.COLOR_RED,     -1),    # Warnings, AI, TURBO
            (CP_INFO,   curses.COLOR_CYAN,    -1),    # Info text, speed indicator
            (CP_DIM,    curses.COLOR_BLACK,   -1),    # Dim (needs A_BOLD for grey)
            (CP_PAD_P,  curses.COLOR_GREEN,   -1),    # Player paddle
            (CP_PAD_AI, curses.COLOR_RED,     -1),    # AI paddle
            (CP_BORDER, curses.COLOR_GREEN,   -1),    # Box-drawing border
            (CP_BALL,   curses.COLOR_WHITE,   -1),    # Ball character
            (CP_TITLE,  curses.COLOR_MAGENTA, -1),    # Intro logo
            (CP_TRAIL,  curses.COLOR_GREEN,   -1),    # Ball trail dots
        ]
        for pid, fg, bg in pairs:
            try:
                curses.init_pair(pid, fg, bg)
            except curses.error:
                pass    # Graceful degradation on limited terminals

    # ── Size check ────────────────────────────────────────────

    def _ok_size(self):
        """Verify the terminal is large enough (60×24 minimum).

        If too small, shows an error and waits for a keypress.
        Returns True if the size is adequate.
        """
        h, w = self.scr.getmaxyx()
        if w < MIN_W or h < MIN_H:
            self.scr.erase()
            m = f'Terminal too small! Need {MIN_W}x{MIN_H}, got {w}x{h}'
            saddstr(self.scr, h // 2, cx(w, len(m)), m,
                    curses.color_pair(CP_DANGER))
            self.scr.refresh()
            self.scr.timeout(-1)     # Block until keypress
            self.scr.getch()
            return False
        return True

    # ── Intro sequence ────────────────────────────────────────

    def _intro(self):
        """Play the intro sequence (skippable with any key).

        Three phases:
          1. Starfield (2 s) — dots scrolling left at varying speeds,
             creating a parallax depth effect.
          2. Logo reveal — "RETRO PONG" characters appear in random
             order, batch by batch, over the starfield.
          3. INSERT COIN — blinking prompt with taglines, waits for
             any key to start the game.

        Music starts at the beginning of phase 1.

        Returns:
            True if the player wants to start, False on size error.
        """
        if not self._ok_size():
            return False
        self.snd.start_music()
        h, w = self.scr.getmaxyx()
        self.scr.timeout(50)     # 20 FPS for intro (less CPU than game)

        # Create 50 random stars with varying speeds (parallax)
        stars = [(random.randint(0, w - 1), random.randint(0, h - 1),
                  random.uniform(0.5, 2.0)) for _ in range(50)]

        def _draw_stars():
            """Move stars leftward and redraw. Stars wrap at the left edge."""
            for i, (sx, sy, sp) in enumerate(stars):
                sx -= sp
                if sx < 0:
                    sx = w - 1
                    sy = random.randint(0, h - 1)
                stars[i] = (sx, sy, sp)
                # Faster stars appear brighter (parallax brightness cue)
                ch = '.' if sp < 1.0 else ('*' if sp < 1.5 else '#')
                saddstr(self.scr, int(sy), int(sx), ch,
                        curses.color_pair(CP_DIM) | curses.A_BOLD)

        # ── Phase 1: Starfield ──
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            self.scr.erase()
            _draw_stars()
            self.scr.refresh()
            if self.scr.getch() != -1:
                return True

        # ── Phase 2: Logo reveal ──
        # Generate the big ASCII-art text for "RETRO" and "PONG"
        logo_retro = text_art('RETRO')
        logo_pong = text_art('PONG')
        logo = logo_retro + [''] + logo_pong
        ly = max(1, (h - len(logo) - 10) // 2)  # Vertical centering

        # Collect all non-space character positions, then shuffle them
        # so the logo appears in a random "materialisation" pattern
        chars = []
        for ri, line in enumerate(logo):
            lx = cx(w, len(line))
            for ci, ch in enumerate(line):
                if ch.strip():
                    chars.append((ly + ri, lx + ci, ch))
        random.shuffle(chars)
        batch = max(1, len(chars) // 15)   # Reveal in ~15 frames

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

        # ── Phase 3: Taglines + INSERT COIN ──
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
            # Blink INSERT COIN at 3 Hz
            if int(dt * 3) % 2:
                saddstr(self.scr, sub_y + 5, cx(w, len(coin)), coin,
                        curses.color_pair(CP_SCORE) | curses.A_BOLD)
            else:
                saddstr(self.scr, sub_y + 5, cx(w, len(coin)),
                        ' ' * len(coin))
            # Show PRESS ANY KEY after 1 second, blinking at 2 Hz
            if dt > 1.0 and int(dt * 2) % 2:
                saddstr(self.scr, sub_y + 7, cx(w, len(prompt)), prompt,
                        curses.color_pair(CP_BRIGHT))
            else:
                saddstr(self.scr, sub_y + 7, cx(w, len(prompt)),
                        ' ' * len(prompt))
            self.scr.refresh()
            if self.scr.getch() != -1:
                return True
            if dt > 30:          # Auto-start after 30 seconds
                return True
        return True

    # ── Field initialisation ──────────────────────────────────

    def _init_field(self):
        """Set up the playing field, paddles, ball, scores, and AI.

        The field dimensions are derived from the current terminal size:
          - Width = terminal_width - 2 (for left/right borders)
          - Height = terminal_height - SCORE_ROWS - 3 (borders + info bar)

        Called at the start of each new game.
        """
        h, w = self.scr.getmaxyx()
        self.th, self.tw = h, w                # Cache terminal dimensions
        self.f_top = SCORE_ROWS + 1            # First row of the game field
        self.f_left = 1                        # First column of the game field
        self.fw = w - 2                        # Field width (inside borders)
        self.fh = h - SCORE_ROWS - 3           # Field height

        # Paddle x-positions (fixed columns near each edge)
        self.px_p = 2                          # Player paddle column
        self.px_ai = self.fw - 3               # AI paddle column

        # Paddle y-positions (start centred vertically)
        self.py_p = self.fh / 2.0 - PADDLE_H / 2
        self.py_ai = self.fh / 2.0 - PADDLE_H / 2

        # Ball state (floating-point for smooth movement)
        self.bx = self.fw / 2.0
        self.by = self.fh / 2.0
        self.bdx = 0.0         # X component of direction (unit vector)
        self.bdy = 0.0         # Y component of direction (unit vector)
        self.bspd = BALL_SPD_INIT
        self.rally = 0         # Hit counter within the current rally
        self.trail = deque(maxlen=3)   # Last 3 ball positions for trail effect

        # Score and game state
        self.sc_p = 0          # Player score
        self.sc_ai = 0         # AI score
        self.ai = AI(self.fw, self.fh)  # Fresh AI opponent
        self.paused = False
        self.rnd = 1           # Round counter (increments on each point)
        self.fx = FX()         # Fresh effects system
        self._serve()          # Initial ball serve

    def _serve(self, d=None):
        """Serve the ball from centre court.

        Resets ball to the centre, picks a random angle within ±45°,
        normalises to a unit direction vector, and resets speed.

        Args:
            d: Serve direction (-1 = left, +1 = right).
               If None, picks randomly.
        """
        self.bx = self.fw / 2.0
        self.by = self.fh / 2.0
        d = d or random.choice([-1, 1])
        a = random.uniform(-math.pi / 4, math.pi / 4)
        self.bdx = math.cos(a) * d
        self.bdy = math.sin(a)
        # Normalise to unit vector (bdx, bdy are direction; bspd is magnitude)
        m = math.hypot(self.bdx, self.bdy)
        if m:
            self.bdx /= m
            self.bdy /= m
        self.bspd = BALL_SPD_INIT
        self.trail.clear()

    # ── Physics ───────────────────────────────────────────────

    def _step_ball(self):
        """Advance the ball by one frame: movement, collisions, scoring.

        Ball movement: bx += bdx * bspd, by += bdy * bspd
        (direction × speed gives actual velocity in cells/frame).

        Collision handling:
          1. Wall bounce: reflect Y velocity, mirror Y position.
          2. Paddle hit: reflect X velocity, adjust Y angle based on
             where the ball hit the paddle (top = steep up, bottom =
             steep down, centre = flat), renormalise to unit vector,
             increase speed by 5 %.
          3. Scoring: if ball passes behind a paddle, award point,
             trigger explosion/shake effects, reset rally, serve.

        Returns:
            'player' or 'ai' if a point was scored, else None.
        """
        # Save position for trail effect
        self.trail.append((self.bx, self.by))

        # Move ball
        self.bx += self.bdx * self.bspd
        self.by += self.bdy * self.bspd

        # ── Wall bounce (top and bottom) ──
        if self.by < 0:
            self.by = -self.by                  # Mirror off top wall
            self.bdy = abs(self.bdy)            # Ensure moving downward
            self.snd.wall()
        elif self.by >= self.fh - 1:
            self.by = 2 * (self.fh - 1) - self.by  # Mirror off bottom wall
            self.bdy = -abs(self.bdy)               # Ensure moving upward
            self.snd.wall()

        # ── Player paddle collision (left side) ──
        # Check: ball moving left, within paddle's x-zone, within paddle's y-zone
        if (self.bdx < 0 and
                self.px_p - 0.5 <= self.bx <= self.px_p + 1.5 and
                self.py_p - 0.5 <= self.by <= self.py_p + PADDLE_H - 0.5):
            self.bdx = abs(self.bdx)            # Reflect: now moving right
            # Hit position determines outgoing angle:
            # hp=0.0 (top of paddle) → steep upward, hp=1.0 → steep downward
            hp = (self.by - self.py_p) / PADDLE_H
            self.bdy = (hp - 0.5) * 2.0        # Range: -1.0 to +1.0
            # Re-normalise to unit vector
            m = math.hypot(self.bdx, self.bdy)
            if m:
                self.bdx /= m; self.bdy /= m
            self.bx = self.px_p + 1.6           # Push ball outside paddle
            self.rally += 1
            self.bspd = min(self.bspd * BALL_SPD_MULT, BALL_SPD_MAX)
            self.snd.hit()

        # ── AI paddle collision (right side) ──
        elif (self.bdx > 0 and
              self.px_ai - 1.5 <= self.bx <= self.px_ai + 0.5 and
              self.py_ai - 0.5 <= self.by <= self.py_ai + PADDLE_H - 0.5):
            self.bdx = -abs(self.bdx)           # Reflect: now moving left
            hp = (self.by - self.py_ai) / PADDLE_H
            self.bdy = (hp - 0.5) * 2.0
            m = math.hypot(self.bdx, self.bdy)
            if m:
                self.bdx /= m; self.bdy /= m
            self.bx = self.px_ai - 1.6          # Push ball outside paddle
            self.rally += 1
            self.bspd = min(self.bspd * BALL_SPD_MULT, BALL_SPD_MAX)
            self.ai.on_hit()
            self.snd.hit()
            # Show "AI EVOLVED!" popup every 3 hits
            if self.ai.hits > 0 and self.ai.hits % 3 == 0:
                self.fx.popup(
                    f'AI EVOLVED! LVL {self.ai.level}',
                    self.f_top + 1,
                    cx(self.tw, 20), 45, CP_DANGER)

        # ── Scoring ──
        scored = None
        if self.bx < -1:
            scored = 'ai'; self.sc_ai += 1      # Ball passed player's side
        elif self.bx > self.fw:
            scored = 'player'; self.sc_p += 1    # Ball passed AI's side

        if scored:
            self.snd.score()
            # Spawn explosion at the side where the ball exited
            ex = 3 if scored == 'ai' else self.fw - 3
            self.fx.explode(self.f_left + ex,
                            self.f_top + int(self.by), 18)
            self.fx.flash = 5        # Brief white flash
            self.fx.shake = 6        # Screen shake for 6 frames
            if scored == 'ai':
                self.ai.on_miss()    # Train AI on the miss experience
            # Pause briefly so the player sees the explosion
            self._draw()
            self.scr.refresh()
            time.sleep(0.6)
            # Reset rally and serve toward the player who was scored on
            self.rally = 0
            self.rnd += 1
            self._serve(1 if scored == 'player' else -1)
            if self.sc_p < WINNING_SCORE and self.sc_ai < WINNING_SCORE:
                self._show_round()
        return scored

    # ── AI update ─────────────────────────────────────────────

    def _step_ai(self):
        """Move the AI paddle one frame and record an experience transition.

        The AI chooses an action (up/stay/down), moves its paddle, then
        records a small shaped reward for DQN training:
          +0.05 if the AI moved closer to the predicted ball landing Y
          -0.02 if it moved further away
          0.0   if the ball is heading away (no useful signal)
        """
        acy = self.py_ai + PADDLE_H / 2   # Current paddle centre
        # Get action from the hybrid policy
        a = self.ai.act(self.bx, self.by,
                        self.bdx * self.bspd, self.bdy * self.bspd, acy)
        # Move paddle
        self.py_ai += self.ai.ACTIONS[a] * self.ai.speed
        self.py_ai = max(0, min(self.fh - PADDLE_H, self.py_ai))

        # Record shaped reward for DQN training
        ncy = self.py_ai + PADDLE_H / 2   # New paddle centre
        ns = self.ai._state(self.bx, self.by,
                            self.bdx * self.bspd, self.bdy * self.bspd, ncy)
        if self.bdx > 0 and self.bspd > 0:
            # Ball heading toward AI: reward for reducing distance to
            # predicted landing position
            steps = max(0, (self.px_ai - self.bx) /
                        max(0.01, self.bdx * self.bspd))
            py = predict_y(self.by, self.bdy * self.bspd, steps, self.fh)
            od = abs(py - acy)     # Old distance to prediction
            nd = abs(py - ncy)     # New distance to prediction
            r = 0.05 if nd < od else -0.02
        else:
            r = 0.0                # Ball going away — no reward signal
        self.ai.record(self.ai.last_s, self.ai.last_a, r, ns, False)

    # ── Drawing ───────────────────────────────────────────────

    def _draw(self):
        """Render one complete frame: borders, scores, field, ball, paddles, effects.

        The screen is fully redrawn every frame (curses erase + redraw pattern).
        The layout from top to bottom:
          Row 0:              Top border (╔═══╗)
          Rows 1–6:           Score area (big digits, labels, divider)
          Row 7:              Separator  (╠═══╣)
          Rows 8–(h-3):       Game field (paddles, ball, center line)
          Row (h-2):          Bottom separator (╠═══╣)
          Row (h-1):          Info bar (controls, AI level) + bottom border
        """
        self.scr.erase()
        h, w = self.th, self.tw
        ba = curses.color_pair(CP_BORDER)

        # ── Top border ──
        saddstr(self.scr, 0, 0, '\u2554' + '\u2550' * (w - 2) + '\u2557', ba)

        # ── Side borders for score area ──
        for y in range(1, SCORE_ROWS):
            saddstr(self.scr, y, 0, '\u2551', ba)       # ║ left
            saddstr(self.scr, y, w - 1, '\u2551', ba)   # ║ right

        # ── Horizontal separator between score area and field ──
        saddstr(self.scr, SCORE_ROWS, 0,
                '\u2560' + '\u2550' * (w - 2) + '\u2563', ba)  # ╠═══╣

        # ── Score display (big LED digits) ──
        self._draw_scores()

        # ── Side borders for game field ──
        for y in range(self.f_top, self.f_top + self.fh):
            saddstr(self.scr, y, 0, '\u2551', ba)
            saddstr(self.scr, y, w - 1, '\u2551', ba)

        # ── Centre court line (dashed vertical) ──
        mid = w // 2
        for y in range(self.f_top, self.f_top + self.fh):
            if y % 2 == 0:   # Every other row for dashed effect
                saddstr(self.scr, y, mid, CH_CENTER,
                        curses.color_pair(CP_DIM) | curses.A_BOLD)

        # ── Bottom separator + info bar ──
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
                    '\u255a' + '\u2550' * (w - 2) + '\u255d', ba)  # ╚═══╝

        # ── Ball trail (fading dots behind the ball) ──
        for i, (tx, ty) in enumerate(self.trail):
            ry = self.f_top + int(ty)
            rx = self.f_left + int(tx)
            ch = CH_TRAIL[min(i, len(CH_TRAIL) - 1)]
            saddstr(self.scr, ry, rx, ch, curses.color_pair(CP_TRAIL))

        # ── Ball ──
        saddstr(self.scr, self.f_top + int(self.by),
                self.f_left + int(self.bx), CH_BALL,
                curses.color_pair(CP_BALL) | curses.A_BOLD)

        # ── Speed indicator (visible from the first rally hit) ──
        # Shows current multiplier with escalating labels and colours:
        #   x1.1         (cyan)    — just started speeding up
        #   x1.5 FAST    (yellow)  — getting challenging
        #   x2.5 TURBO!  (red)     — nearly unplayable
        if self.rally > 0:
            mult = self.bspd / BALL_SPD_INIT
            if mult >= 2.5:
                t = f'x{mult:.1f} TURBO!'
                cp = CP_DANGER
            elif mult >= 1.5:
                t = f'x{mult:.1f} FAST'
                cp = CP_SCORE
            else:
                t = f'x{mult:.1f}'
                cp = CP_INFO
            saddstr(self.scr, self.f_top, cx(w, len(t)), t,
                    curses.color_pair(cp) | curses.A_BOLD)

        # ── Paddles ──
        for py in range(PADDLE_H):
            # Player paddle (green, left side)
            saddstr(self.scr, self.f_top + int(self.py_p) + py,
                    self.f_left + self.px_p, CH_PAD,
                    curses.color_pair(CP_PAD_P) | curses.A_BOLD)
            # AI paddle (red, right side)
            saddstr(self.scr, self.f_top + int(self.py_ai) + py,
                    self.f_left + self.px_ai, CH_PAD,
                    curses.color_pair(CP_PAD_AI) | curses.A_BOLD)

        # ── Particle effects and popup messages ──
        self.fx.render(self.scr)

        # ── Pause overlay (centred box) ──
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
        """Render the score area: player/CPU labels, centre divider, big digits.

        The score area occupies rows 1–6 and is split vertically by a
        thin line.  Each half shows the respective player's score as
        large 3×5 block digits.
        """
        w = self.tw
        mid = w // 2
        # Labels
        saddstr(self.scr, 1, 2, 'PLAYER',
                curses.color_pair(CP_PAD_P) | curses.A_BOLD)
        saddstr(self.scr, 1, w - 5, 'CPU',
                curses.color_pair(CP_PAD_AI) | curses.A_BOLD)
        # Centre divider line
        saddstr(self.scr, 1, mid, '\u2502',
                curses.color_pair(CP_DIM) | curses.A_BOLD)
        for y in range(2, SCORE_ROWS):
            saddstr(self.scr, y, mid, '\u2502',
                    curses.color_pair(CP_DIM) | curses.A_BOLD)

        # Big LED-style digit scores, centred in each half
        ps = str(self.sc_p)
        pw = len(ps) * 4 - 1          # Width of rendered digits
        self._big_num(self.sc_p, 2, mid // 2 - pw // 2, CP_PAD_P)
        ai_s = str(self.sc_ai)
        aw = len(ai_s) * 4 - 1
        self._big_num(self.sc_ai, 2, mid + (w - mid) // 2 - aw // 2,
                      CP_PAD_AI)

    def _big_num(self, num, y, x, cp):
        """Render a number as large 3×5 block digits.

        Each digit is 3 columns wide with a 1-column gap between digits.
        Uses the DIGITS lookup table.

        Args:
            num: Integer to render.
            y, x: Top-left position for the first digit.
            cp:   Colour pair ID.
        """
        for ch in str(num):
            g = DIGITS.get(ch, DIGITS['0'])
            for ri, row in enumerate(g):
                saddstr(self.scr, y + ri, x, row,
                        curses.color_pair(cp) | curses.A_BOLD)
            x += 4   # 3 chars + 1 space

    def _show_round(self):
        """Display a "ROUND N" banner for 1 second between points.

        Renders the current game frame behind the banner so the
        player can see the score while waiting.
        """
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

    # ── Game loop ─────────────────────────────────────────────

    def _loop(self):
        """Main game loop: input → AI → physics → render at 30 FPS.

        Uses a fixed-timestep approach:
          1. Drain all pending keyboard input (non-blocking).
          2. If not paused: advance AI, advance ball, update effects.
          3. Render the frame.
          4. Sleep for the remainder of the frame budget (1/30 s).

        The loop continues until one player reaches WINNING_SCORE.

        Returns:
            'player' or 'ai' indicating who won.
        """
        self.scr.timeout(0)   # Non-blocking getch
        while self.sc_p < WINNING_SCORE and self.sc_ai < WINNING_SCORE:
            t0 = time.monotonic()

            # Drain all pending key events (responsive even during
            # fast key repeat / held keys)
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

            # Sleep for remaining frame time to hit 30 FPS target
            dt = time.monotonic() - t0
            rem = (1.0 / FPS) - dt
            if rem > 0.001:
                time.sleep(rem)

        return 'player' if self.sc_p >= WINNING_SCORE else 'ai'

    def _key(self, k):
        """Handle a single keypress.

        Q = quit (raises KeyboardInterrupt for clean exit).
        P = toggle pause.
        M = toggle music.
        W/S/Up/Down = move player paddle (only when not paused).
        """
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
            # Clamp paddle within field bounds
            self.py_p = max(0, min(self.fh - PADDLE_H, self.py_p))

    # ── Win / Lose screens ────────────────────────────────────

    def _win_screen(self):
        """Show victory animation: big "YOU WIN!" text, ASCII trophy, fireworks.

        The fireworks are particle bursts spawned at random positions.
        Runs for 6 seconds or until a key is pressed (after 1.5 s).
        """
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
        fx = FX()   # Separate effects system for this screen
        while time.monotonic() - t0 < 6.0:
            self.scr.erase()
            t = time.monotonic() - t0
            # Colour-cycling title
            cp = CP_SCORE if int(t * 4) % 2 else CP_BRIGHT
            ty = 2
            for row in title_rows:
                saddstr(self.scr, ty, cx(w, len(row)), row,
                        curses.color_pair(cp) | curses.A_BOLD)
                ty += 1
            # Trophy
            ty += 1
            for line in trophy:
                saddstr(self.scr, ty, cx(w, len(line)), line,
                        curses.color_pair(CP_SCORE) | curses.A_BOLD)
                ty += 1
            # Final score and AI level
            sc = f'FINAL SCORE: {self.sc_p} - {self.sc_ai}'
            saddstr(self.scr, ty + 1, cx(w, len(sc)), sc,
                    curses.color_pair(CP_BRIGHT))
            al = f'AI LEVEL REACHED: {self.ai.level}'
            saddstr(self.scr, ty + 2, cx(w, len(al)), al,
                    curses.color_pair(CP_INFO))
            # Random firework bursts
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
        """Show defeat animation: big "GAME OVER" with glitch/static effects.

        The glitch effect randomly shifts text lines horizontally and
        overlays scanline-style noise.  Runs for 5 seconds.
        """
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
            # Colour flicker between red and white
            cp = CP_DANGER if int(t * 6) % 3 != 0 else CP_BRIGHT
            # "GAME" text with random horizontal glitch
            for i, row in enumerate(go_rows):
                off = random.randint(-2, 2) if random.random() < 0.12 else 0
                saddstr(self.scr, gy + i, cx(w, len(row)) + off, row,
                        curses.color_pair(cp) | curses.A_BOLD)
            # "OVER" text
            for i, row in enumerate(ov_rows):
                off = random.randint(-2, 2) if random.random() < 0.12 else 0
                saddstr(self.scr, gy + 6 + i, cx(w, len(row)) + off, row,
                        curses.color_pair(cp) | curses.A_BOLD)
            # Final score
            sc = f'FINAL SCORE: {self.sc_p} - {self.sc_ai}'
            saddstr(self.scr, gy + 13, cx(w, len(sc)), sc,
                    curses.color_pair(CP_BRIGHT))
            # Random static noise line (CRT distortion effect)
            if random.random() < 0.3:
                ny = random.randint(0, h - 1)
                ns = ''.join(random.choice('\u2591\u2592\u2593\u2588 ')
                             for _ in range(w))    # ░▒▓█ and space
                saddstr(self.scr, ny, 0, ns,
                        curses.color_pair(CP_DIM) | curses.A_BOLD)
            self.scr.refresh()
            if self.scr.getch() != -1 and t > 1.5:
                break

    # ── Highscore screen ──────────────────────────────────────

    def _hs_screen(self):
        """Show highscore entry (if qualified) and the full score table.

        Score formula: player_points × 100 + ai_level × 10.
        This rewards both winning and pushing the AI to higher levels.
        """
        self.scr.timeout(120)
        h, w = self.th, self.tw
        final = self.sc_p * 100 + self.ai.level * 10
        hi = -1   # Index of the newly inserted entry (-1 = none)
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

        # Display the full table until a key is pressed
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

    # ── Play again prompt ─────────────────────────────────────

    def _again(self):
        """Ask the player to play again (Y/N).

        The prompt blinks at 2 Hz.  Returns True for Y, False for N/Q/Esc.
        """
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

    # ── Main flow ─────────────────────────────────────────────

    def run(self):
        """Top-level game flow: intro → (game → end → scores → again?) loop.

        The outer try/except ensures that KeyboardInterrupt (Q key or
        Ctrl+C) exits cleanly, and the finally block kills any lingering
        audio subprocesses.
        """
        try:
            if not self._intro():
                return
            while True:
                self._init_field()
                # Resume music if it was stopped (e.g. after a previous game)
                if not self.snd._music_proc:
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
    """Entry point called by curses.wrapper.

    curses.wrapper handles terminal setup (raw mode, no echo) and
    teardown (restore terminal on exit or exception).
    """
    RetroPong(stdscr).run()


if __name__ == '__main__':
    # Reduce the ESC key delay from the default 1000 ms to 25 ms.
    # This makes the Escape key (used for arrow-key sequences) feel
    # responsive instead of laggy.
    os.environ.setdefault('ESCDELAY', '25')
    curses.wrapper(main)
