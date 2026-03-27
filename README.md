# RETRO PONG

> 80s arcade-style ASCII Pong with a machine learning opponent that evolves as you play.

```
╔════════════════════════════════════════════════════════════╗
║ PLAYER         │          CPU                              ║
║  ███           │            █                              ║
║    █           │          ███                              ║
║  ███           │          █                                ║
║  █             │          ███                              ║
║  ███           │                                           ║
╠════════════════════════════════════════════════════════════╣
║█               ┊                                         █║
║█               ┊              ●                          █║
║█               ┊                                         █║
║█               ┊                                         █║
╠════════════════════════════════════════════════════════════╣
║ W/S:Move  P:Pause  M:Music  Q:Quit          AI LVL:3     ║
╚════════════════════════════════════════════════════════════╝
```

## Features

- **Retro ASCII graphics** with Unicode box-drawing, big LED-style score digits, ball trail, and center court line
- **ML opponent** using a neural network (Deep Q-Learning) that starts competent and genuinely improves with every hit
- **Chiptune music** - procedurally generated square/triangle wave Amiga-style soundtrack
- **Sound effects** - hit, wall bounce, score, win/lose jingles
- **Intro sequence** - starfield animation, character-by-character logo reveal, "INSERT COIN" prompt
- **Particle effects** - explosions on score, screen shake, turbo indicator
- **Win/Lose animations** - trophy + fireworks for wins, glitch/static GAME OVER for losses
- **Arcade highscore table** - top 10, classic 3-letter initial entry, persistent across games
- **Game plays to 11 points**

## How the AI Works

The opponent uses a hybrid machine learning approach:

1. **Baseline**: A trajectory-prediction heuristic ensures ~95%+ hit rate from the start
2. **Neural Network**: A 2-layer MLP (6 inputs, 128 hidden ReLU, 3 outputs) is pre-trained on 5000 synthetic demonstrations
3. **Live Learning**: Deep Q-Learning with experience replay refines the network during gameplay
4. **Gradual Takeover**: The NN's influence increases with each successful hit (visible as "AI LEVEL")
5. **Result**: The AI starts solid and gets noticeably faster and more precise over time

## Controls

| Key | Action |
|-----|--------|
| `W` / `S` or `Up` / `Down` | Move paddle |
| `P` | Pause |
| `M` | Toggle music |
| `Q` | Quit |

## Installation

### Run directly (requires Python 3.8+ and NumPy)

```bash
pip install numpy
python3 pong.py
```

### Docker

```bash
docker build -t retro-pong .
docker run -it retro-pong
```

> **Note:** Sound requires ALSA and may not work inside Docker without audio passthrough.

### Portable zipapp

```bash
./build_zip.sh
python3 retro-pong.pyz
```

## Requirements

- Python 3.8+
- NumPy
- Terminal with UTF-8 support (minimum 60x24)
- `aplay` for sound (optional, Linux - game works silently without it)

## License

MIT
