#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

_ROOT_DIR   = Path(__file__).parent.parent.parent
_MAX_DIR    = _ROOT_DIR / "Max"
_SOUNDS_DIR = _ROOT_DIR / "Sounds"

# demo mode just copies these into Max/ as 1.wav, 2.wav, etc
_DEMO_MAP = {
    "Synth.wav": "1.wav",
    "Bass.wav":  "2.wav",
    "Drums.wav": "3.wav",
    "Riser.wav": "4.wav",
}

# shutting up mediapipe/tensorflow log spam, took forever to figure out
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# the C++ libs print straight to fd 2 ignoring python's sys.stderr
# so we have to dup2 it to /dev/null. python tracebacks still work because
# we save the original fd and re-attach sys.stderr to it
_orig_stderr_fd = os.dup(2)
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)
os.close(_devnull_fd)
sys.stderr = os.fdopen(_orig_stderr_fd, "w", closefd=True)

sys.path.insert(0, str(Path(__file__).parent))

from views.menu import run_menu, MenuChoice
from views.demo_mix import run_mix
from views.choose_sound import run_choose_sound
from views.tutorial import run_tutorial
from core.gui import WIN_W, WIN_H


def _placeholder(title: str):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

    font = cv2.FONT_HERSHEY_SIMPLEX
    print(f"[{title}] — press Q or ESC to return to the menu.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        overlay = np.zeros_like(frame)
        cv2.addWeighted(frame, 0.4, overlay, 0.6, 0, frame)

        msg = f"{title}"
        (tw, _), _ = cv2.getTextSize(msg, font, 1.4, 2)
        cv2.putText(
            frame,
            msg,
            ((fw - tw) // 2, fh // 2 - 20),
            font,
            1.4,
            (120, 210, 255),
            2,
            cv2.LINE_AA,
        )

        sub = "Coming soon  |  Q / ESC = back to menu"
        (sw, _), _ = cv2.getTextSize(sub, font, 0.55, 1)
        cv2.putText(
            frame,
            sub,
            ((fw - sw) // 2, fh // 2 + 30),
            font,
            0.55,
            (90, 150, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("MOTION MIX", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> int:
    while True:
        choice = run_menu()

        if choice == MenuChoice.QUIT:
            print("Shaka!")
            return 0

        elif choice == MenuChoice.DEMO:
            _MAX_DIR.mkdir(parents=True, exist_ok=True)
            for src_name, dst_name in _DEMO_MAP.items():
                src = _SOUNDS_DIR / src_name
                if src.exists():
                    shutil.copy2(src, _MAX_DIR / dst_name)
            run_mix(num_channels=4, names=["Synth", "Bass", "Drums", "Riser"])

        elif choice == MenuChoice.SOUND:
            run_choose_sound()

        elif choice == MenuChoice.TUTORIAL:
            run_tutorial()


if __name__ == "__main__":
    sys.exit(main())
