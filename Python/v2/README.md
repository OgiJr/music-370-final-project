# Motion Mix v2

Gesture-controlled audio mixer. Your webcam tracks your hands in real time and sends OSC messages over UDP to Max/MSP, which applies them to a 4-channel audio engine. No MIDI controllers, no knobs, just your hands.

Built with MediaPipe (hand tracking), OpenCV (UI + camera), and Python. The Max patch handles all the actual audio.

---

## How it works

```
webcam → Python (MediaPipe detects hands) → UDP/OSC → Max/MSP (audio engine)
```

Python figures out what your hands are doing and converts that into OSC messages like `/1_vol 72` or `/2_eq_freq 58`. Max listens on port 9000, receives those, and adjusts the audio in real time. Both have to be running at the same time on the same machine.

---

## Setup

Commands 1 and 3 are required only the first time you run the app

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 main.py
```

Then open `Max/motion-mix.maxpat` in Max/MSP and make sure DSP is on (`Cmd + /` or the speaker icon). **Check that your sample rate is 48000 Hz** in Options → Audio Status or everything will sound pitch-shifted and wrong.

---

## Gestures

| Gesture | What it does |
|---|---|
| Open palm (right hand) | Move cursor / aim at stuff |
| Fist + hold ~0.9s | Activate whatever you're hovering |
| Pinch (thumb + index) | Grab a volume fader, drag up/down |
| Twist (thumb + middle + rotate wrist) | Pan left/right |
| Shaka 🤙 hold 0.8s | Go back / quit |
| Left hand N fingers held 1.5s | Open channel N's effects screen |

In the effects screen, left hand switches tabs (1 = Stereo Field, 2 = Reverb, 3 = EQ).

---

## Modes

**Try out Demo** — loads the bundled Sounds/ folder wavs and drops you straight into the mixer. Good for testing that everything works.

**Choose Your Sound** — lets you pick 4 of your own WAV files (one per channel). fist-hold each slot to open a file browser, then fist-hold START MIX. Keys 1–4 also open the picker if the gesture is being annoying.

**Tutorial** — 5 slides covering setup, gestures, and every screen. Arrow keys to navigate.

---

## Project structure

```
Python/v2/
├── main.py                  entry point, handles mode routing
├── core/
│   ├── hand_detection.py    MediaPipe wrapper, all gesture logic
│   ├── gui.py               shared colors, drawing helpers, buttons
│   └── udp.py               OSC packet builder + UDP sender
└── views/
    ├── menu.py              main menu screen
    ├── demo_mix.py          the actual mixer (faders + knobs)
    ├── choose_sound.py      file upload screen
    ├── tutorial.py          tutorial slides
    └── channel_mods/        per-channel effects screens
        ├── stereo_field.py  XY pad for pan + volume
        ├── reverb.py        room size + decay sliders
        └── eq.py            biquad EQ with live frequency plot
```

---

## Troubleshooting

**Mediapipe spam in the terminal** — already suppressed, but if it comes back set `GLOG_minloglevel=3` and `TF_CPP_MIN_LOG_LEVEL=3` before running.

**Gestures feel laggy** — lower the camera resolution in `core/gui.py` (`WIN_W`, `WIN_H`). 1920×1080 assumes a decent machine.

**Max isn't responding** — make sure DSP is on and the patch is actually open. UDP just fires and forgets, Python won't tell you if nothing is listening.

**Pinch and twist keep interfering** — that's mostly expected, they share overlapping finger geometry. The code has hysteresis and mutual exclusion but it's not perfect.
