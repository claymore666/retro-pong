# RETRO PONG — 80s/90s Arcade ASCII Pong with ML Opponent

## Context
Full-featured retro arcade Pong game with ML opponent, intro sequence, animations, chiptune music, arcade highscore table. Packaged as Docker container + zipapp, published as public GitHub repo.

## File Structure
```
pong/
├── pong.py              # Main game (single entry point)
├── Dockerfile           # Docker container
├── requirements.txt     # Just numpy
├── README.md            # Project README with screenshots/usage
├── .gitignore
└── build_zip.sh         # Script to create portable zipapp
```

## Features

### 1. Intro Sequence / "Game Trailer" (~10 seconds, skippable)
- Large ASCII art "RETRO PONG" logo with typing/reveal animation
- Scrolling starfield background effect
- "INSERT COIN" blinking text
- Quick demo: auto-playing pong rally for 3-4 seconds
- "PRESS ANY KEY TO START" with fade-in
- Credits line: "POWERED BY MACHINE LEARNING"

### 2. Retro 80s/90s Graphics (curses + Unicode)
- Double-line border (`╔═╗║╚╝`) with corner decorations
- Scanline-style horizontal lines (subtle dimming on alternate rows)
- Dashed center court line
- Big 3x5 pixel-font score numbers (LED-style)
- Paddles: thick `█` blocks (height 4)
- Ball: `●` with motion trail (fading dots behind it)
- "ROUND X" display between points
- Color scheme: green-on-black (classic CRT) or amber option
- Screen flash on score

### 3. Animations & Gimmicks
- **Ball trail**: 2-3 fading characters behind ball showing direction
- **Score explosion**: ASCII particle burst when someone scores (`* . · ✦`)
- **Screen shake**: Brief offset on score (simulated via padding)
- **"GOAL!"** text pops up with expanding box animation
- **Win screen**: Large ASCII trophy + fireworks animation (ascending `*` that burst into patterns)
- **Lose screen**: "GAME OVER" with dripping/melting text effect
- **Speed-up warning**: "⚡TURBO⚡" flashes when ball gets fast
- **AI level-up**: Brief "AI EVOLVED!" flash with level indicator

### 4. Chiptune Music & Sound
- Generate square/triangle wave music with numpy → play via `aplay`
- Background loop: simple chiptune melody (C major arpeggio pattern)
- Hit sound: short blip
- Score sound: descending buzz
- Win/lose jingle
- M key to toggle music on/off

### 5. ML Opponent (Deep Q-Learning)
- 2-layer MLP: 5 inputs → 64 hidden (ReLU) → 3 outputs
- Pre-trained on 500 synthetic ball trajectories (90%+ baseline)
- Live learning from replay buffer after each hit
- Visible "AI LEVEL" meter that increases
- AI paddle speed increases slightly with level
- Epsilon: 0.1 → 0.01 (less random over time)

### 6. Arcade Highscore Table
- Shown after game ends (first to 11 points wins)
- Top 10 scores, stored in `~/.retropong_scores.json`
- Classic "ENTER YOUR INITIALS" — 3 characters, arcade style letter selector (↑↓ to pick letter, → to confirm)
- Table shows: RANK | INITIALS | SCORE | DATE
- Highlight new entry with blinking

### 7. Game Flow
1. Intro sequence (skippable with any key)
2. Main menu: "1P VS AI" (only mode for now)
3. Game plays to 11 points
4. Win/Lose animation
5. Highscore entry if qualified
6. Highscore table display
7. "PLAY AGAIN? Y/N"

### 8. Controls
- W/S or ↑/↓: Move paddle
- M: Toggle music
- P: Pause (with retro "PAUSED" overlay)
- Q: Quit
- Any key: Skip intro

### 9. Packaging

**Docker:**
```dockerfile
FROM python:3.11-slim
RUN pip install numpy
COPY pong.py /app/pong.py
WORKDIR /app
ENV TERM=xterm-256color
CMD ["python3", "pong.py"]
```
Run: `docker run -it retro-pong`

**Zipapp (build_zip.sh):**
Creates a self-contained `pong.pyz` that runs with `python3 pong.pyz` (requires numpy installed).

### 10. GitHub Repository
- Create public repo `retro-pong` under claymore666's GitHub
- README with: ASCII screenshot, feature list, installation, Docker usage, controls
- Use `gh repo create` to create and push
