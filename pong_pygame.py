#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════╗
║      N E O N   P O N G                    ║
║  Synthwave Arcade  ·  ML Opponent         ║
╚═══════════════════════════════════════════╝

Pygame port of Retro Pong with neon glow visuals.
"""

import math
import random
import time
import json
import sys
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

WIDTH, HEIGHT = 1280, 720
FPS = 60
WINNING_SCORE = 11

# Field area (inside borders)
BORDER = 4
SCORE_AREA_H = 100
FIELD_TOP = SCORE_AREA_H + BORDER
FIELD_LEFT = BORDER
FIELD_W = WIDTH - 2 * BORDER
FIELD_H = HEIGHT - SCORE_AREA_H - 2 * BORDER - 40  # 40 for bottom info bar
INFO_BAR_Y = HEIGHT - 40

PADDLE_W = 14
PADDLE_H = 90
PADDLE_MARGIN = 30

BALL_RADIUS = 10
BALL_SPD_INIT = 5.0
BALL_SPD_MULT = 1.05
BALL_SPD_MAX = 25.0
PLAYER_SPD = 7.0
AI_SPD_BASE = 4.5

# ═══════════════════════════════════════════════════════════════
#  COLORS — synthwave palette
# ═══════════════════════════════════════════════════════════════

BG_COLOR = (8, 4, 20)
BG_FIELD = (12, 6, 28)
NEON_CYAN = (0, 255, 255)
NEON_MAGENTA = (255, 0, 200)
NEON_PINK = (255, 40, 120)
NEON_PURPLE = (180, 60, 255)
NEON_YELLOW = (255, 255, 60)
NEON_ORANGE = (255, 140, 20)
NEON_GREEN = (0, 255, 140)
NEON_WHITE = (255, 255, 255)
NEON_RED = (255, 30, 60)
DIM_PURPLE = (60, 20, 80)
DIM_CYAN = (0, 80, 80)
GRID_COLOR = (25, 12, 50)
SCANLINE_COLOR = (0, 0, 0)

PLAYER_COLOR = NEON_CYAN
AI_COLOR = NEON_MAGENTA
BALL_COLOR = NEON_WHITE
BORDER_COLOR = NEON_PURPLE


# ═══════════════════════════════════════════════════════════════
#  GLOW / POST-PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════

def make_glow_surface(w, h, color, radius=None, intensity=1.0):
    """Create a radial glow surface for additive blending."""
    if radius is None:
        radius = max(w, h)
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    cx, cy = radius, radius
    for r in range(radius, 0, -1):
        alpha = int(intensity * 60 * (r / radius) ** 0.5 * (1 - r / radius) ** 0.8)
        alpha = max(0, min(255, alpha))
        c = (*color[:3], alpha)
        pygame.draw.circle(surf, c, (cx, cy), r)
    return surf


def draw_glowing_rect(surface, color, rect, glow_radius=20, intensity=1.0):
    """Draw a rectangle with neon glow."""
    x, y, w, h = rect
    # Outer glow
    glow = pygame.Surface((w + glow_radius * 2, h + glow_radius * 2), pygame.SRCALPHA)
    for i in range(glow_radius, 0, -2):
        alpha = int(intensity * 80 * (1 - i / glow_radius) ** 1.5)
        alpha = max(0, min(255, alpha))
        c = (*color[:3], alpha)
        pygame.draw.rect(glow, c,
                         (glow_radius - i, glow_radius - i,
                          w + i * 2, h + i * 2),
                         border_radius=max(2, i // 2))
    surface.blit(glow, (x - glow_radius, y - glow_radius),
                 special_flags=pygame.BLEND_ADD)
    # Solid core
    pygame.draw.rect(surface, color, rect, border_radius=3)
    # Bright highlight
    highlight = (*[min(255, c + 80) for c in color[:3]],)
    inner = (x + 2, y + 2, max(1, w - 4), max(1, h - 4))
    pygame.draw.rect(surface, highlight, inner, border_radius=2)


def draw_glowing_circle(surface, color, pos, radius, glow_radius=25, intensity=1.0):
    """Draw a circle with neon glow."""
    glow = make_glow_surface(radius, radius, color, glow_radius, intensity)
    surface.blit(glow, (pos[0] - glow_radius, pos[1] - glow_radius),
                 special_flags=pygame.BLEND_ADD)
    pygame.draw.circle(surface, color, pos, radius)
    highlight = tuple(min(255, c + 100) for c in color[:3])
    pygame.draw.circle(surface, highlight, pos, max(1, radius - 3))


def draw_glowing_line(surface, color, start, end, width=2, glow_width=8, intensity=0.5):
    """Draw a line with glow effect."""
    for i in range(glow_width, 0, -2):
        alpha = int(intensity * 60 * (1 - i / glow_width))
        alpha = max(0, min(255, alpha))
        c = (*color[:3], alpha)
        glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(glow_surf, c, start, end, width + i * 2)
        surface.blit(glow_surf, (0, 0), special_flags=pygame.BLEND_ADD)
    pygame.draw.line(surface, color, start, end, width)


# ═══════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════

class Particle:
    __slots__ = ('x', 'y', 'dx', 'dy', 'color', 'life', 'max_life', 'size')

    def __init__(self, x, y, dx, dy, color, life=1.0, size=4):
        self.x = float(x)
        self.y = float(y)
        self.dx = dx
        self.dy = dy
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size


class FX:
    def __init__(self):
        self.particles = []
        self.shake_amount = 0
        self.shake_offset = (0, 0)
        self.flash_alpha = 0
        self.messages = []  # (text, x, y, ttl, color, size)
        self.trail_points = []  # (x, y, age, color)

    def explode(self, x, y, n=30, colors=None):
        if colors is None:
            colors = [NEON_CYAN, NEON_MAGENTA, NEON_PINK, NEON_YELLOW, NEON_WHITE]
        for _ in range(n):
            a = random.uniform(0, 2 * math.pi)
            v = random.uniform(1.0, 8.0)
            self.particles.append(Particle(
                x, y,
                math.cos(a) * v, math.sin(a) * v,
                random.choice(colors),
                random.uniform(0.5, 1.5),
                random.randint(2, 6),
            ))

    def spark(self, x, y, dx, dy, n=8, color=NEON_WHITE):
        for _ in range(n):
            a = random.uniform(0, 2 * math.pi)
            v = random.uniform(0.5, 3.0)
            self.particles.append(Particle(
                x, y,
                dx * 0.5 + math.cos(a) * v,
                dy * 0.5 + math.sin(a) * v,
                color, random.uniform(0.2, 0.6), random.randint(2, 4),
            ))

    def popup(self, text, x, y, ttl=90, color=NEON_WHITE, size=36):
        self.messages.append([text, x, y, ttl, color, size])

    def add_trail(self, x, y, color=BALL_COLOR):
        self.trail_points.append([x, y, 0, color])

    def update(self, dt):
        alive = []
        for p in self.particles:
            p.x += p.dx * dt * 60
            p.y += p.dy * dt * 60
            p.dy += 0.15 * dt * 60
            p.life -= dt
            if p.life > 0:
                alive.append(p)
        self.particles = alive

        if self.shake_amount > 0:
            self.shake_amount *= 0.85
            if self.shake_amount < 0.5:
                self.shake_amount = 0
                self.shake_offset = (0, 0)
            else:
                self.shake_offset = (
                    random.randint(int(-self.shake_amount), int(self.shake_amount)),
                    random.randint(int(-self.shake_amount), int(self.shake_amount)),
                )
        else:
            self.shake_offset = (0, 0)

        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 8)

        self.messages = [
            [t, x, y - dt * 30, ttl - 1, c, s]
            for t, x, y, ttl, c, s in self.messages if ttl > 1
        ]

        new_trail = []
        for tp in self.trail_points:
            tp[2] += dt
            if tp[2] < 0.4:
                new_trail.append(tp)
        self.trail_points = new_trail

    def render(self, surface, font_cache):
        # Trail
        for tx, ty, age, color in self.trail_points:
            alpha = max(0, 1.0 - age / 0.4)
            r = int(BALL_RADIUS * 1.5 * alpha)
            if r > 0:
                glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
                c = (*color[:3], int(alpha * 80))
                pygame.draw.circle(glow, c, (r * 2, r * 2), r * 2)
                c2 = (*color[:3], int(alpha * 160))
                pygame.draw.circle(glow, c2, (r * 2, r * 2), r)
                surface.blit(glow, (int(tx) - r * 2, int(ty) - r * 2),
                             special_flags=pygame.BLEND_ADD)

        # Particles
        for p in self.particles:
            alpha = p.life / p.max_life
            r = max(1, int(p.size * alpha))
            glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            c = (*p.color[:3], int(alpha * 200))
            pygame.draw.circle(glow, c, (r * 2, r * 2), r * 2)
            surface.blit(glow, (int(p.x) - r * 2, int(p.y) - r * 2),
                         special_flags=pygame.BLEND_ADD)
            pygame.draw.circle(surface, p.color, (int(p.x), int(p.y)), max(1, r // 2))

        # Flash
        if self.flash_alpha > 0:
            flash = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            flash.fill((255, 255, 255, self.flash_alpha))
            surface.blit(flash, (0, 0))

        # Messages
        for text, x, y, ttl, color, size in self.messages:
            alpha = min(255, ttl * 6)
            font = font_cache.get(size)
            if font:
                rendered = font.render(text, True, color)
                rendered.set_alpha(alpha)
                rect = rendered.get_rect(center=(int(x), int(y)))
                surface.blit(rendered, rect)


# ═══════════════════════════════════════════════════════════════
#  SOUND ENGINE (chiptune via numpy + pygame.mixer)
# ═══════════════════════════════════════════════════════════════

class Sound:
    RATE = 44100

    def __init__(self):
        self.on = True
        try:
            pygame.mixer.init(frequency=self.RATE, size=-16, channels=1, buffer=512)
            self._ok = True
        except Exception:
            self._ok = False
        self._music_channel = None

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

    def _make_sound(self, samples):
        if not self._ok:
            return None
        # pygame.mixer expects (n, 1) for mono
        buf = np.column_stack([samples]).copy(order='C')
        return pygame.sndarray.make_sound(buf)

    def _play_fx(self, samples):
        if not self._ok or not self.on:
            return
        snd = self._make_sound(samples)
        if snd:
            snd.play()

    def hit(self):
        self._play_fx(self._sq(880, 0.05, 0.12))

    def wall(self):
        self._play_fx(self._sq(440, 0.03, 0.08))

    def score(self):
        self._play_fx(np.concatenate([
            self._sq(600, 0.08, 0.15),
            self._sq(400, 0.08, 0.15),
            self._sq(200, 0.16, 0.15),
        ]))

    def win(self):
        self._play_fx(np.concatenate([
            self._sq(f, 0.10, 0.18)
            for f in [523, 587, 659, 784, 880, 1047]
        ]))

    def lose(self):
        self._play_fx(np.concatenate([
            self._sq(f, 0.13, 0.18)
            for f in [400, 350, 300, 250, 200]
        ]))

    def start_music(self):
        if not self._ok or not self.on:
            return
        self.stop_music()
        samples = self._make_music()
        snd = self._make_sound(samples)
        if snd:
            self._music_channel = snd.play(loops=-1)

    def stop_music(self):
        if self._music_channel:
            self._music_channel.stop()
            self._music_channel = None

    def toggle(self):
        self.on = not self.on
        if self.on:
            self.start_music()
        else:
            self.stop_music()

    def cleanup(self):
        self.stop_music()

    def _make_music(self):
        bpm = 140
        n8 = 60.0 / bpm / 2
        s8 = int(self.RATE * n8)

        C2=65;D2=73;Eb2=78;F2=87;G2=98
        Ab2=104;Bb2=117;B2=123;C3=131;D3=147
        Eb3=156;F3=175;G3=196;Ab3=208;Bb3=233
        B3=247;C4=262;D4=294;Eb4=311;F4=349
        G4=392;Ab4=415;Bb4=466;B4=494;C5=523
        D5=587;Eb5=659;F5=698;G5=784
        REST=0

        def _render(mel, bass_notes, hat_pat, lead_vol=0.10,
                    bass_vol=0.18, hat_vol=0.04, note_len=0.85):
            n = len(mel)
            total = s8 * n
            lead = np.zeros(total, dtype=np.int16)
            bass = np.zeros(total, dtype=np.int16)
            hh = np.zeros(total, dtype=np.int16)
            for i, f in enumerate(mel):
                off = i * s8
                if f > 0:
                    s = self._sq(f, n8 * note_len, lead_vol)
                    e = min(off + len(s), total)
                    lead[off:e] = s[:e - off]
            for i, f in enumerate(bass_notes):
                off = i * s8 * 4
                if f > 0:
                    s = self._tri(f, n8 * 3.9, bass_vol)
                    e = min(off + len(s), total)
                    bass[off:e] = s[:e - off]
            for i in range(n):
                pi = i % len(hat_pat)
                if hat_pat[pi]:
                    off = i * s8
                    s = self._noise(n8 * 0.25, hat_vol)
                    e = min(off + len(s), total)
                    hh[off:e] = s[:e - off]
            return np.clip(
                lead.astype(np.int32) + bass.astype(np.int32) + hh.astype(np.int32),
                -32767, 32767
            ).astype(np.int16)

        intro_mel = [REST]*32
        intro_bass = [C2,C2,C2,C2,Ab2,Ab2,G2,G2]
        intro_hat = [1,0,0,0,1,0,0,0]
        intro = _render(intro_mel, intro_bass, intro_hat, hat_vol=0.05)

        va_mel = [
            C4,Eb4,G4,C5,G4,Eb4,C4,Eb4, C4,G4,C5,G4,Eb4,C4,Eb4,G4,
            Bb3,D4,F4,Bb4,F4,D4,Bb3,D4, Bb3,F4,Bb4,F4,D4,Bb3,D4,F4,
            Ab3,C4,Eb4,Ab4,Eb4,C4,Ab3,C4, Ab3,Eb4,Ab4,Eb4,C4,Ab3,C4,Eb4,
            G3,B3,D4,G4,D4,B3,G3,B3, G3,D4,G4,B4,G4,D4,C4,Eb4,
        ]
        va_bass = [C2,C2,C2,C3,Bb2,Bb2,Bb2,Bb2,Ab2,Ab2,Ab2,Ab2,G2,G2,G2,C2]
        va_hat = [1,0,1,0,1,0,1,0]
        verse_a = _render(va_mel, va_bass, va_hat)

        va2_mel = [
            C5,G4,Eb4,C4,Eb4,G4,C5,G4, C4,G4,C5,G4,Eb4,C4,Eb4,G4,
            Bb4,F4,D4,Bb3,D4,F4,Bb4,F4, Bb3,F4,Bb4,F4,D4,Bb3,D4,F4,
            Ab3,C4,Eb4,Ab4,Eb4,C4,Ab3,C4, Ab3,Eb4,Ab4,Eb4,C4,Ab3,C4,Eb4,
            G3,B3,D4,G4,D4,B3,G3,B3, G3,D4,G4,B4,G4,Eb4,C4,G4,
        ]
        va2_bass = [C2,C2,C2,C3,Bb2,Bb2,Bb2,Bb2,Ab2,Ab2,Ab2,Ab2,G2,G2,G2,C2]
        va2_hat = [1,0,1,0,1,0,1,1]
        verse_a2 = _render(va2_mel, va2_bass, va2_hat)

        ch_mel = [
            C4,Eb4,G4,C5,G4,Eb4,C4,G4, C4,Eb4,G4,C5,Eb5,C5,G4,Eb4,
            Bb3,D4,F4,Bb4,F4,D4,Bb3,F4, Bb3,D4,F4,Bb4,D5,Bb4,F4,D4,
            Ab3,C4,Eb4,Ab4,Eb4,C4,Ab3,Eb4, Ab3,C4,Eb4,Ab4,C5,Ab4,Eb4,C4,
            G3,B3,D4,G4,D4,B3,G4,B4, G4,B4,D5,G5,D5,B4,C5,Eb5,
        ]
        ch_bass = [C2,C2,C2,C3,Bb2,Bb2,Bb2,Bb2,Ab2,Ab2,Ab2,Ab2,G2,G2,G2,C3]
        ch_hat = [1,0,1,1,1,0,1,1]
        chorus = _render(ch_mel, ch_bass, ch_hat, lead_vol=0.12, hat_vol=0.05)

        br_mel = [
            C4,C4,Eb4,Eb4,G4,G4,C5,C5, G4,G4,Eb4,Eb4,C4,C4,C4,C4,
            Bb3,Bb3,D4,D4,F4,F4,Bb4,Bb4, F4,F4,D4,D4,Bb3,Bb3,Bb3,Bb3,
            Ab3,Ab3,C4,C4,Eb4,Eb4,Ab4,Ab4, Eb4,Eb4,C4,C4,Ab3,Ab3,Ab3,Ab3,
            G3,G3,B3,B3,D4,D4,G4,G4, G4,G4,B4,B4,D4,D4,C4,Eb4,
        ]
        br_bass = [C2,C2,C2,C2,Bb2,Bb2,Bb2,Bb2,Ab2,Ab2,Ab2,Ab2,G2,G2,G2,C2]
        br_hat = [1,0,0,0,1,0,0,0]
        bridge = _render(br_mel, br_bass, br_hat,
                         lead_vol=0.09, bass_vol=0.20, note_len=0.95)

        reprise = _render(va_mel, va_bass, va_hat)

        ou_mel = [
            C5,REST,G4,REST,Eb4,REST,C4,REST,
            G4,REST,Eb4,REST,C4,REST,REST,REST,
            Eb4,REST,C4,REST,G3,REST,REST,REST,
            C4,REST,REST,REST,REST,REST,REST,REST,
        ]
        ou_bass = [C2,Ab2,F2,Eb2,Ab2,G2,G2,C2]
        ou_hat = [1,0,0,0,0,0,0,0]
        outro = _render(ou_mel, ou_bass, ou_hat, lead_vol=0.08, hat_vol=0.03)

        return np.concatenate([intro, verse_a, verse_a2, chorus, bridge, reprise, outro])


# ═══════════════════════════════════════════════════════════════
#  NEURAL NETWORK (2-layer MLP)
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
#  AI OPPONENT
# ═══════════════════════════════════════════════════════════════

def predict_y(by, bdy, steps, fh):
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


class AI:
    ACTIONS = [-1, 0, 1]

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
        self.nn_mix = 0.0
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

    def _rule(self, bx, by, bdx, bdy, acy):
        lvl = self.level
        if bdx <= 0.01:
            if lvl < 5:
                return 1
            ty = self.fh / 2.0
            diff = ty - acy
            return 0 if diff < -2 else (2 if diff > 2 else 1)
        react_x = self.fw * max(0.0, 0.65 - lvl * 0.08)
        if bx < react_x:
            return 1
        hesitate = max(0.0, 0.12 - lvl * 0.015)
        if random.random() < hesitate:
            return 1
        steps = max(1, (self.fw - 3 - bx) / bdx)
        if lvl <= 2:
            ty = by
        elif lvl <= 4:
            ty = by + bdy * steps
            ty = max(0.0, min(float(self.fh - 1), ty))
        else:
            ty = predict_y(by, bdy, steps, self.fh)
        noise = max(0.3, 4.0 - lvl * 0.4)
        ty += random.gauss(0, noise)
        track = max(0.0, 0.6 - lvl * 0.08)
        ty = ty * (1.0 - track) + by * track
        ty = max(0.0, min(float(self.fh - 1), ty))
        deadzone = max(0.6, 2.5 - lvl * 0.2)
        diff = ty - acy
        if diff < -deadzone:
            return 0
        if diff > deadzone:
            return 2
        return 1

    def _pretrain(self):
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
        return min(7.5, 3.5 + self.level * 0.3)


# ═══════════════════════════════════════════════════════════════
#  HIGH SCORES
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


# ═══════════════════════════════════════════════════════════════
#  MAIN GAME
# ═══════════════════════════════════════════════════════════════

class NeonPong:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("NEON PONG — Synthwave Arcade")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.snd = Sound()
        self.fx = FX()
        self.hs = HiScores()
        self._init_fonts()
        self._init_scanlines()

    def _init_fonts(self):
        self.fonts = {}
        for size in [16, 20, 24, 28, 32, 36, 48, 64, 72, 96]:
            self.fonts[size] = pygame.font.SysFont('monospace', size, bold=True)

    def _init_scanlines(self):
        self.scanline_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 3):
            pygame.draw.line(self.scanline_surf, (0, 0, 0, 25),
                             (0, y), (WIDTH, y), 1)

    # ── CRT vignette ────────────────────────────────────────
    def _draw_vignette(self, surface):
        vig = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        cx, cy = WIDTH // 2, HEIGHT // 2
        max_r = math.hypot(cx, cy)
        for r in range(int(max_r), int(max_r * 0.5), -3):
            alpha = int(120 * ((r - max_r * 0.5) / (max_r * 0.5)) ** 2)
            alpha = max(0, min(120, alpha))
            pygame.draw.circle(vig, (0, 0, 0, alpha), (cx, cy), r, 3)
        surface.blit(vig, (0, 0))

    # ── perspective grid ────────────────────────────────────
    def _draw_grid(self, surface):
        # Horizontal grid lines in field area
        for y in range(FIELD_TOP, FIELD_TOP + FIELD_H, 30):
            pygame.draw.line(surface, GRID_COLOR,
                             (FIELD_LEFT, y), (FIELD_LEFT + FIELD_W, y), 1)
        # Vertical grid lines
        for x in range(FIELD_LEFT, FIELD_LEFT + FIELD_W, 40):
            pygame.draw.line(surface, GRID_COLOR,
                             (x, FIELD_TOP), (x, FIELD_TOP + FIELD_H), 1)

    # ── draw border ─────────────────────────────────────────
    def _draw_border(self, surface):
        rect = (FIELD_LEFT, FIELD_TOP, FIELD_W, FIELD_H)
        draw_glowing_rect(surface, BORDER_COLOR, rect, glow_radius=12, intensity=0.6)
        # Erase interior so glow is only the frame
        inner = (FIELD_LEFT + 2, FIELD_TOP + 2, FIELD_W - 4, FIELD_H - 4)
        pygame.draw.rect(surface, BG_FIELD, inner)

    # ── center line ─────────────────────────────────────────
    def _draw_center_line(self, surface):
        mid_x = WIDTH // 2
        seg_h = 16
        gap = 12
        y = FIELD_TOP + 4
        while y < FIELD_TOP + FIELD_H - 4:
            end_y = min(y + seg_h, FIELD_TOP + FIELD_H - 4)
            c = (*NEON_PURPLE[:3],)
            pygame.draw.line(surface, DIM_PURPLE, (mid_x, y), (mid_x, end_y), 2)
            # subtle glow
            glow_surf = pygame.Surface((20, end_y - y + 20), pygame.SRCALPHA)
            pygame.draw.line(glow_surf, (*DIM_PURPLE, 40),
                             (10, 0), (10, end_y - y), 8)
            surface.blit(glow_surf, (mid_x - 10, y - 2), special_flags=pygame.BLEND_ADD)
            y += seg_h + gap

    # ── draw scores ─────────────────────────────────────────
    def _draw_scores(self, surface, sc_p, sc_ai, ai_level):
        # Score area background
        pygame.draw.rect(surface, BG_COLOR, (0, 0, WIDTH, SCORE_AREA_H))

        # Separator line
        draw_glowing_line(surface, BORDER_COLOR,
                          (BORDER, SCORE_AREA_H), (WIDTH - BORDER, SCORE_AREA_H),
                          width=2, glow_width=6, intensity=0.4)

        # Labels
        label_font = self.fonts[20]
        p_label = label_font.render("PLAYER", True, PLAYER_COLOR)
        surface.blit(p_label, (40, 8))
        ai_label = label_font.render("CPU", True, AI_COLOR)
        surface.blit(ai_label, (WIDTH - 90, 8))

        # Big score numbers
        score_font = self.fonts[72]
        p_score = score_font.render(str(sc_p), True, PLAYER_COLOR)
        # Glow behind player score
        glow = score_font.render(str(sc_p), True, PLAYER_COLOR)
        glow_surf = pygame.Surface(glow.get_size(), pygame.SRCALPHA)
        glow_surf.blit(glow, (0, 0))
        glow_surf.set_alpha(60)
        for ox, oy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            surface.blit(glow_surf, (WIDTH // 4 - p_score.get_width() // 2 + ox,
                                     25 + oy), special_flags=pygame.BLEND_ADD)
        surface.blit(p_score, (WIDTH // 4 - p_score.get_width() // 2, 25))

        ai_score = score_font.render(str(sc_ai), True, AI_COLOR)
        glow2 = score_font.render(str(sc_ai), True, AI_COLOR)
        glow_surf2 = pygame.Surface(glow2.get_size(), pygame.SRCALPHA)
        glow_surf2.blit(glow2, (0, 0))
        glow_surf2.set_alpha(60)
        for ox, oy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            surface.blit(glow_surf2, (3 * WIDTH // 4 - ai_score.get_width() // 2 + ox,
                                      25 + oy), special_flags=pygame.BLEND_ADD)
        surface.blit(ai_score, (3 * WIDTH // 4 - ai_score.get_width() // 2, 25))

        # Center divider
        pygame.draw.line(surface, DIM_PURPLE,
                         (WIDTH // 2, 8), (WIDTH // 2, SCORE_AREA_H - 8), 2)

    # ── draw info bar ───────────────────────────────────────
    def _draw_info_bar(self, surface, ai_level, rally, bspd):
        pygame.draw.rect(surface, BG_COLOR, (0, INFO_BAR_Y, WIDTH, 40))
        draw_glowing_line(surface, BORDER_COLOR,
                          (BORDER, INFO_BAR_Y), (WIDTH - BORDER, INFO_BAR_Y),
                          width=2, glow_width=6, intensity=0.4)

        info_font = self.fonts[20]
        controls = info_font.render("W/S: Move   P: Pause   M: Music   Q: Quit",
                                    True, DIM_PURPLE)
        surface.blit(controls, (20, INFO_BAR_Y + 10))

        ai_text = f"AI LVL: {ai_level}"
        ai_surf = info_font.render(ai_text, True, NEON_RED)
        surface.blit(ai_surf, (WIDTH - ai_surf.get_width() - 20, INFO_BAR_Y + 10))

        # Speed indicator
        if rally > 0:
            mult = bspd / BALL_SPD_INIT
            if mult >= 2.5:
                label = f"x{mult:.1f} TURBO!"
                color = NEON_RED
            elif mult >= 1.5:
                label = f"x{mult:.1f} FAST"
                color = NEON_YELLOW
            else:
                label = f"x{mult:.1f}"
                color = NEON_GREEN
            spd_surf = info_font.render(label, True, color)
            surface.blit(spd_surf, (WIDTH // 2 - spd_surf.get_width() // 2,
                                    INFO_BAR_Y + 10))

    # ── intro sequence ──────────────────────────────────────
    def _intro(self):
        self.snd.start_music()
        stars = [(random.randint(0, WIDTH), random.randint(0, HEIGHT),
                  random.uniform(1.0, 5.0), random.randint(1, 3))
                 for _ in range(120)]
        t0 = time.monotonic()
        phase = 0  # 0=stars, 1=logo reveal, 2=prompts
        logo_text = "NEON PONG"
        revealed_chars = 0
        reveal_timer = 0

        while True:
            dt = self.clock.tick(FPS) / 1000.0
            t = time.monotonic() - t0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    return True

            self.screen.fill(BG_COLOR)

            # Starfield
            for i, (sx, sy, sp, sz) in enumerate(stars):
                sx -= sp * dt * 60
                if sx < 0:
                    sx = WIDTH
                    sy = random.randint(0, HEIGHT)
                stars[i] = (sx, sy, sp, sz)
                brightness = int(min(255, sp * 60))
                c = (brightness, brightness, brightness)
                pygame.draw.circle(self.screen, c, (int(sx), int(sy)), sz)

            if phase == 0 and t > 1.5:
                phase = 1

            if phase >= 1:
                reveal_timer += dt
                if reveal_timer > 0.06:
                    reveal_timer = 0
                    revealed_chars = min(revealed_chars + 1, len(logo_text))

                title_font = self.fonts[96]
                if revealed_chars > 0:
                    partial = logo_text[:revealed_chars]
                    # Glow
                    title_surf = title_font.render(partial, True, NEON_CYAN)
                    glow_surf = pygame.Surface(title_surf.get_size(), pygame.SRCALPHA)
                    glow_surf.blit(title_surf, (0, 0))
                    glow_surf.set_alpha(80)
                    tx = WIDTH // 2 - title_font.size(logo_text)[0] // 2
                    ty = HEIGHT // 3 - 40
                    for ox, oy in [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, -4), (0, 4)]:
                        self.screen.blit(glow_surf, (tx + ox, ty + oy),
                                         special_flags=pygame.BLEND_ADD)
                    self.screen.blit(title_surf, (tx, ty))

                if revealed_chars >= len(logo_text) and phase < 2:
                    phase = 2

            if phase >= 2:
                sub_font = self.fonts[24]
                sub = sub_font.render("── SYNTHWAVE ARCADE  ·  ML OPPONENT ──",
                                      True, NEON_PURPLE)
                self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2,
                                       HEIGHT // 3 + 60))

                ml = self.fonts[20].render("POWERED BY MACHINE LEARNING",
                                           True, DIM_PURPLE)
                self.screen.blit(ml, (WIDTH // 2 - ml.get_width() // 2,
                                      HEIGHT // 3 + 100))

                # INSERT COIN blink
                if int(t * 3) % 2:
                    coin = self.fonts[32].render(">>> INSERT COIN <<<",
                                                 True, NEON_YELLOW)
                    self.screen.blit(coin, (WIDTH // 2 - coin.get_width() // 2,
                                            HEIGHT * 2 // 3))

                if t > 4.0 and int(t * 2) % 2:
                    prompt = self.fonts[24].render("PRESS ANY KEY TO START",
                                                   True, NEON_WHITE)
                    self.screen.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2,
                                              HEIGHT * 2 // 3 + 60))

            self.screen.blit(self.scanline_surf, (0, 0))
            pygame.display.flip()

            if t > 30:
                return True

    # ── field init ──────────────────────────────────────────
    def _init_field(self):
        self.py_p = FIELD_TOP + FIELD_H / 2 - PADDLE_H / 2
        self.py_ai = FIELD_TOP + FIELD_H / 2 - PADDLE_H / 2
        self.px_p = FIELD_LEFT + PADDLE_MARGIN
        self.px_ai = FIELD_LEFT + FIELD_W - PADDLE_MARGIN - PADDLE_W

        self.bx = WIDTH / 2.0
        self.by = FIELD_TOP + FIELD_H / 2.0
        self.bdx = 0.0
        self.bdy = 0.0
        self.bspd = BALL_SPD_INIT
        self.rally = 0

        self.sc_p = 0
        self.sc_ai = 0
        self.ai = AI(FIELD_W, FIELD_H)
        self.paused = False
        self.rnd = 1
        self.fx = FX()
        self._serve()

    def _serve(self, d=None):
        self.bx = WIDTH / 2.0
        self.by = FIELD_TOP + FIELD_H / 2.0
        d = d or random.choice([-1, 1])
        a = random.uniform(-math.pi / 4, math.pi / 4)
        self.bdx = math.cos(a) * d
        self.bdy = math.sin(a)
        m = math.hypot(self.bdx, self.bdy)
        if m:
            self.bdx /= m
            self.bdy /= m
        self.bspd = BALL_SPD_INIT

    # ── physics ─────────────────────────────────────────────
    def _step_ball(self, dt):
        self.fx.add_trail(self.bx, self.by, BALL_COLOR)

        self.bx += self.bdx * self.bspd * dt * 60
        self.by += self.bdy * self.bspd * dt * 60

        # Wall bounce
        if self.by - BALL_RADIUS < FIELD_TOP:
            self.by = FIELD_TOP + BALL_RADIUS
            self.bdy = abs(self.bdy)
            self.snd.wall()
            self.fx.spark(self.bx, self.by, 0, 1, 5, BORDER_COLOR)
        elif self.by + BALL_RADIUS > FIELD_TOP + FIELD_H:
            self.by = FIELD_TOP + FIELD_H - BALL_RADIUS
            self.bdy = -abs(self.bdy)
            self.snd.wall()
            self.fx.spark(self.bx, self.by, 0, -1, 5, BORDER_COLOR)

        # Player paddle collision
        p_rect = pygame.Rect(self.px_p, self.py_p, PADDLE_W, PADDLE_H)
        if (self.bdx < 0 and
                p_rect.left - BALL_RADIUS <= self.bx <= p_rect.right and
                p_rect.top - BALL_RADIUS <= self.by <= p_rect.bottom + BALL_RADIUS):
            self.bdx = abs(self.bdx)
            hp = (self.by - self.py_p) / PADDLE_H
            self.bdy = (hp - 0.5) * 2.0
            m = math.hypot(self.bdx, self.bdy)
            if m:
                self.bdx /= m
                self.bdy /= m
            self.bx = p_rect.right + BALL_RADIUS + 1
            self.rally += 1
            self.bspd = min(self.bspd * BALL_SPD_MULT, BALL_SPD_MAX)
            self.snd.hit()
            self.fx.spark(self.bx, self.by, 1, 0, 10, PLAYER_COLOR)

        # AI paddle collision
        ai_rect = pygame.Rect(self.px_ai, self.py_ai, PADDLE_W, PADDLE_H)
        if (self.bdx > 0 and
                ai_rect.left <= self.bx + BALL_RADIUS and
                self.bx <= ai_rect.right + BALL_RADIUS and
                ai_rect.top - BALL_RADIUS <= self.by <= ai_rect.bottom + BALL_RADIUS):
            self.bdx = -abs(self.bdx)
            hp = (self.by - self.py_ai) / PADDLE_H
            self.bdy = (hp - 0.5) * 2.0
            m = math.hypot(self.bdx, self.bdy)
            if m:
                self.bdx /= m
                self.bdy /= m
            self.bx = ai_rect.left - BALL_RADIUS - 1
            self.rally += 1
            self.bspd = min(self.bspd * BALL_SPD_MULT, BALL_SPD_MAX)
            self.ai.on_hit()
            self.snd.hit()
            self.fx.spark(self.bx, self.by, -1, 0, 10, AI_COLOR)
            if self.ai.hits > 0 and self.ai.hits % 3 == 0:
                self.fx.popup(f"AI EVOLVED! LVL {self.ai.level}",
                              WIDTH // 2, FIELD_TOP + 30, 90, NEON_RED, 28)

        # Scoring
        scored = None
        if self.bx < FIELD_LEFT - 20:
            scored = 'ai'
            self.sc_ai += 1
        elif self.bx > FIELD_LEFT + FIELD_W + 20:
            scored = 'player'
            self.sc_p += 1

        if scored:
            self.snd.score()
            ex = self.px_p if scored == 'ai' else self.px_ai
            self.fx.explode(ex, self.by, 40,
                            [PLAYER_COLOR if scored == 'player' else AI_COLOR,
                             NEON_WHITE, NEON_YELLOW])
            self.fx.flash_alpha = 80
            self.fx.shake_amount = 15
            if scored == 'ai':
                self.ai.on_miss()
            self.rally = 0
            self.rnd += 1
            self._serve(1 if scored == 'player' else -1)
            if self.sc_p < WINNING_SCORE and self.sc_ai < WINNING_SCORE:
                self._show_round()
        return scored

    # ── AI update ───────────────────────────────────────────
    def _step_ai(self, dt):
        # Convert ball pos to field-relative coords for AI
        rel_bx = self.bx - FIELD_LEFT
        rel_by = self.by - FIELD_TOP
        acy = self.py_ai - FIELD_TOP + PADDLE_H / 2
        a = self.ai.act(rel_bx, rel_by,
                        self.bdx * self.bspd, self.bdy * self.bspd, acy)
        self.py_ai += self.ai.ACTIONS[a] * self.ai.speed * dt * 60
        self.py_ai = max(FIELD_TOP, min(FIELD_TOP + FIELD_H - PADDLE_H, self.py_ai))

        # Record experience
        ncy = self.py_ai - FIELD_TOP + PADDLE_H / 2
        ns = self.ai._state(rel_bx, rel_by,
                            self.bdx * self.bspd, self.bdy * self.bspd, ncy)
        if self.bdx > 0 and self.bspd > 0:
            rel_px_ai = self.px_ai - FIELD_LEFT
            steps = max(0, (rel_px_ai - rel_bx) / max(0.01, self.bdx * self.bspd))
            py = predict_y(rel_by, self.bdy * self.bspd, steps, FIELD_H)
            od = abs(py - acy)
            nd = abs(py - ncy)
            r = 0.05 if nd < od else -0.02
        else:
            r = 0.0
        self.ai.record(self.ai.last_s, self.ai.last_a, r, ns, False)

    # ── draw frame ──────────────────────────────────────────
    def _draw(self):
        surface = self.screen
        surface.fill(BG_COLOR)

        # Grid
        self._draw_grid(surface)

        # Border
        self._draw_border(surface)

        # Center line
        self._draw_center_line(surface)

        # Scores
        self._draw_scores(surface, self.sc_p, self.sc_ai, self.ai.level)

        # Info bar
        self._draw_info_bar(surface, self.ai.level, self.rally, self.bspd)

        # Effects (trail + particles under paddles/ball)
        self.fx.render(surface, self.fonts)

        # Paddles
        draw_glowing_rect(surface, PLAYER_COLOR,
                          (self.px_p, self.py_p, PADDLE_W, PADDLE_H),
                          glow_radius=18, intensity=0.8)
        draw_glowing_rect(surface, AI_COLOR,
                          (self.px_ai, self.py_ai, PADDLE_W, PADDLE_H),
                          glow_radius=18, intensity=0.8)

        # Ball
        draw_glowing_circle(surface, BALL_COLOR,
                            (int(self.bx), int(self.by)),
                            BALL_RADIUS, glow_radius=30, intensity=1.2)

        # Pause overlay
        if self.paused:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            surface.blit(overlay, (0, 0))
            pause_font = self.fonts[64]
            pause_text = pause_font.render("PAUSED", True, NEON_CYAN)
            surface.blit(pause_text,
                         (WIDTH // 2 - pause_text.get_width() // 2,
                          HEIGHT // 2 - pause_text.get_height() // 2))

        # Scanlines + vignette (post-processing)
        surface.blit(self.scanline_surf, self.fx.shake_offset)
        self._draw_vignette(surface)

    def _show_round(self):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.2:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            self._draw()
            # Round overlay
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self.screen.blit(overlay, (0, 0))

            round_text = f"ROUND {self.rnd}"
            font = self.fonts[48]
            rendered = font.render(round_text, True, NEON_YELLOW)
            # Glow
            glow = font.render(round_text, True, NEON_YELLOW)
            glow.set_alpha(80)
            cx = WIDTH // 2 - rendered.get_width() // 2
            cy = HEIGHT // 2 - rendered.get_height() // 2
            for ox, oy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
                self.screen.blit(glow, (cx + ox, cy + oy),
                                 special_flags=pygame.BLEND_ADD)
            self.screen.blit(rendered, (cx, cy))
            pygame.display.flip()

    # ── game loop ───────────────────────────────────────────
    def _loop(self):
        while self.sc_p < WINNING_SCORE and self.sc_ai < WINNING_SCORE:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)  # cap delta time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        raise KeyboardInterrupt
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    if event.key == pygame.K_m:
                        self.snd.toggle()

            if not self.paused:
                # Continuous key input for smooth movement
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    self.py_p -= PLAYER_SPD * dt * 60
                if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                    self.py_p += PLAYER_SPD * dt * 60
                self.py_p = max(FIELD_TOP,
                                min(FIELD_TOP + FIELD_H - PADDLE_H, self.py_p))

                self._step_ai(dt)
                self._step_ball(dt)
                self.fx.update(dt)

            self._draw()
            pygame.display.flip()

        return 'player' if self.sc_p >= WINNING_SCORE else 'ai'

    # ── win screen ──────────────────────────────────────────
    def _win_screen(self):
        self.snd.win()
        t0 = time.monotonic()
        firework_fx = FX()

        while time.monotonic() - t0 < 6.0:
            dt = self.clock.tick(FPS) / 1000.0
            t = time.monotonic() - t0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and t > 1.5:
                    return

            self.screen.fill(BG_COLOR)

            # Title
            font = self.fonts[96]
            cp = NEON_YELLOW if int(t * 4) % 2 else NEON_CYAN
            title = font.render("YOU WIN!", True, cp)
            glow = font.render("YOU WIN!", True, cp)
            glow.set_alpha(80)
            tx = WIDTH // 2 - title.get_width() // 2
            ty = 80
            for ox, oy in [(-4, -4), (4, -4), (-4, 4), (4, 4)]:
                self.screen.blit(glow, (tx + ox, ty + oy),
                                 special_flags=pygame.BLEND_ADD)
            self.screen.blit(title, (tx, ty))

            # Trophy (simple geometric)
            cx, cy = WIDTH // 2, HEIGHT // 2 - 20
            # Cup body
            trophy_color = NEON_YELLOW
            points = [(cx - 60, cy - 50), (cx + 60, cy - 50),
                      (cx + 40, cy + 30), (cx - 40, cy + 30)]
            pygame.draw.polygon(self.screen, trophy_color, points, 3)
            # Handles
            pygame.draw.arc(self.screen, trophy_color,
                            (cx - 85, cy - 45, 30, 50), -1.5, 1.5, 3)
            pygame.draw.arc(self.screen, trophy_color,
                            (cx + 55, cy - 45, 30, 50), 1.6, 4.7, 3)
            # Base
            pygame.draw.rect(self.screen, trophy_color,
                             (cx - 15, cy + 30, 30, 20), 2)
            pygame.draw.rect(self.screen, trophy_color,
                             (cx - 35, cy + 50, 70, 8), 2)

            # Score
            score_font = self.fonts[32]
            sc = score_font.render(f"FINAL SCORE: {self.sc_p} - {self.sc_ai}",
                                   True, NEON_WHITE)
            self.screen.blit(sc, (WIDTH // 2 - sc.get_width() // 2, cy + 80))
            al = self.fonts[24].render(f"AI LEVEL REACHED: {self.ai.level}",
                                       True, NEON_CYAN)
            self.screen.blit(al, (WIDTH // 2 - al.get_width() // 2, cy + 120))

            # Fireworks
            if random.random() < 0.15:
                fx_x = random.randint(100, WIDTH - 100)
                fx_y = random.randint(50, HEIGHT // 2)
                firework_fx.explode(fx_x, fx_y, 25,
                                    [NEON_CYAN, NEON_MAGENTA, NEON_YELLOW,
                                     NEON_GREEN, NEON_PINK])

            firework_fx.update(dt)
            firework_fx.render(self.screen, self.fonts)
            self.screen.blit(self.scanline_surf, (0, 0))
            pygame.display.flip()

    # ── lose screen ─────────────────────────────────────────
    def _lose_screen(self):
        self.snd.lose()
        t0 = time.monotonic()

        while time.monotonic() - t0 < 5.0:
            dt = self.clock.tick(FPS) / 1000.0
            t = time.monotonic() - t0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and t > 1.5:
                    return

            self.screen.fill(BG_COLOR)

            # Glitch effect
            font = self.fonts[96]
            cp = NEON_RED if int(t * 6) % 3 != 0 else NEON_WHITE

            game_text = font.render("GAME", True, cp)
            over_text = font.render("OVER", True, cp)

            # Shake on text
            ox1 = random.randint(-4, 4) if random.random() < 0.15 else 0
            ox2 = random.randint(-4, 4) if random.random() < 0.15 else 0
            gy = HEIGHT // 3

            self.screen.blit(game_text,
                             (WIDTH // 2 - game_text.get_width() // 2 + ox1, gy))
            self.screen.blit(over_text,
                             (WIDTH // 2 - over_text.get_width() // 2 + ox2,
                              gy + 100))

            # Score
            sc = self.fonts[32].render(
                f"FINAL SCORE: {self.sc_p} - {self.sc_ai}", True, NEON_WHITE)
            self.screen.blit(sc, (WIDTH // 2 - sc.get_width() // 2, gy + 220))

            # Static noise
            if random.random() < 0.3:
                ny = random.randint(0, HEIGHT - 3)
                noise_surf = pygame.Surface((WIDTH, 3), pygame.SRCALPHA)
                for x in range(0, WIDTH, 4):
                    v = random.randint(20, 80)
                    noise_surf.fill((v, v, v, 80), (x, 0, 4, 3))
                self.screen.blit(noise_surf, (0, ny))

            self.screen.blit(self.scanline_surf, (0, 0))
            pygame.display.flip()

    # ── highscore screen ────────────────────────────────────
    def _hs_screen(self):
        final = self.sc_p * 100 + self.ai.level * 10
        hi = -1

        if self.hs.qualifies(final):
            initials = self._enter_initials()
            if initials:
                hi = self.hs.add(initials, final, self.ai.level)

        t0 = time.monotonic()
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            t = time.monotonic() - t0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and t > 0.8:
                    return

            self.screen.fill(BG_COLOR)

            # Title
            title_font = self.fonts[36]
            title = title_font.render("═══ HIGH SCORES ═══", True, NEON_YELLOW)
            self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 60))

            # Header
            hdr_font = self.fonts[24]
            hdr = hdr_font.render(" #   NAME   SCORE   LVL   DATE", True, NEON_CYAN)
            self.screen.blit(hdr, (WIDTH // 2 - hdr.get_width() // 2, 120))

            # Line
            pygame.draw.line(self.screen, DIM_PURPLE,
                             (WIDTH // 4, 150), (3 * WIDTH // 4, 150), 1)

            # Entries
            entry_font = self.fonts[24]
            for i, e in enumerate(self.hs.data):
                line = (f"{i+1:>2}.  {e['initials']:<5} {e['score']:>6}   "
                        f"{e.get('ai_level', '?'):>3}   {e.get('date', ''):>10}")
                if i == hi:
                    color = NEON_YELLOW if int(t * 3) % 2 else NEON_WHITE
                else:
                    color = DIM_PURPLE if i % 2 else NEON_PURPLE
                rendered = entry_font.render(line, True, color)
                self.screen.blit(rendered,
                                 (WIDTH // 2 - rendered.get_width() // 2, 165 + i * 32))

            if not self.hs.data:
                no = entry_font.render("NO SCORES YET!", True, DIM_PURPLE)
                self.screen.blit(no, (WIDTH // 2 - no.get_width() // 2, HEIGHT // 2))

            prompt = self.fonts[20].render("PRESS ANY KEY", True, DIM_PURPLE)
            self.screen.blit(prompt, (WIDTH // 2 - prompt.get_width() // 2, HEIGHT - 80))

            self.screen.blit(self.scanline_surf, (0, 0))
            pygame.display.flip()

    def _enter_initials(self):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        sel = [0, 0, 0]
        pos = 0

        while True:
            dt = self.clock.tick(FPS) / 1000.0
            t = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_w, pygame.K_UP):
                        sel[pos] = (sel[pos] - 1) % len(letters)
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        sel[pos] = (sel[pos] + 1) % len(letters)
                    elif event.key in (pygame.K_RIGHT, pygame.K_TAB,
                                       pygame.K_RETURN, pygame.K_KP_ENTER):
                        if pos < 2:
                            pos += 1
                        else:
                            return ''.join(letters[i] for i in sel)
                    elif event.key == pygame.K_LEFT and pos > 0:
                        pos -= 1

            self.screen.fill(BG_COLOR)

            title = self.fonts[36].render("** NEW HIGH SCORE! **", True, NEON_YELLOW)
            self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

            sub = self.fonts[24].render("ENTER YOUR INITIALS", True, NEON_CYAN)
            self.screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, 160))

            # Letter slots
            letter_font = self.fonts[64]
            total_w = 3 * 60 + 2 * 30
            start_x = WIDTH // 2 - total_w // 2
            cy = HEIGHT // 2

            for i in range(3):
                ch = letters[sel[i]]
                lx = start_x + i * 90
                color = NEON_WHITE if i == pos else DIM_PURPLE

                if i == pos and int(t * 4) % 2:
                    # Highlight box
                    draw_glowing_rect(self.screen, NEON_CYAN,
                                      (lx - 5, cy - 40, 70, 80),
                                      glow_radius=10, intensity=0.5)

                rendered = letter_font.render(ch, True, color)
                self.screen.blit(rendered, (lx + 15, cy - 30))

                # Underline
                pygame.draw.line(self.screen, color,
                                 (lx, cy + 45), (lx + 60, cy + 45), 3)

                # Arrows for current position
                if i == pos:
                    arrow_up = self.fonts[24].render("▲", True, NEON_CYAN)
                    self.screen.blit(arrow_up, (lx + 20, cy - 65))
                    arrow_dn = self.fonts[24].render("▼", True, NEON_CYAN)
                    self.screen.blit(arrow_dn, (lx + 20, cy + 55))

            hint = self.fonts[20].render("UP/DOWN: SELECT   RIGHT/ENTER: NEXT",
                                         True, DIM_PURPLE)
            self.screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, cy + 120))

            self.screen.blit(self.scanline_surf, (0, 0))
            pygame.display.flip()

    # ── play again ──────────────────────────────────────────
    def _again(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            t = time.monotonic()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_y,):
                        return True
                    if event.key in (pygame.K_n, pygame.K_q, pygame.K_ESCAPE):
                        return False

            if int(t * 2) % 2:
                prompt = self.fonts[32].render("PLAY AGAIN? (Y/N)", True, NEON_YELLOW)
                self.screen.blit(prompt,
                                 (WIDTH // 2 - prompt.get_width() // 2,
                                  HEIGHT - 80))
            pygame.display.flip()

    # ── main flow ───────────────────────────────────────────
    def run(self):
        try:
            if not self._intro():
                return
            while True:
                self._init_field()
                if not self.snd._music_channel:
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
            pygame.quit()


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    NeonPong().run()
