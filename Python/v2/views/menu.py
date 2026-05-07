import os
import time
from enum import Enum

os.environ["GLOG_minloglevel"] = os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2

from core.hand_detection import HandDetector
from core.gui import (
    dark_overlay, draw_button, draw_title, draw_cursor, flash,
    C_PLUM, C_MAGENTA, C_AMBER, C_WHITE,
    HOLD_SECS, FONT, WIN_W, WIN_H,
)


class MenuChoice(Enum):
    DEMO     = "demo"
    SOUND    = "sound"
    TUTORIAL = "tutorial"
    QUIT     = "quit"


BTN_W, BTN_H, BTN_GAP = 250, 200, 50

BUTTONS = [
    {"label": "Try out Demo",      "icon": "play", "id": MenuChoice.DEMO},
    {"label": "Choose Your Sound", "icon": "note", "id": MenuChoice.SOUND},
    {"label": "Tutorial",          "icon": "info", "id": MenuChoice.TUTORIAL},
]

SHAKA_HOLD = 0.8


def _draw_shaka_hint(
    frame, fw: int, fh: int, progress: float
) -> None:
    label = "hold shaka to quit"
    col   = C_AMBER if progress > 0 else C_PLUM
    scale = 0.52
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
    tx = (fw - tw) // 2
    ty = fh - 14
    cv2.putText(frame, label, (tx, ty), FONT, scale, col, 1, cv2.LINE_AA)
    if progress > 0:
        cx = tx + tw + 20
        cv2.ellipse(frame, (cx, ty - th // 2), (10, 10),
                    -90, 0, int(360 * progress), C_AMBER, 2, cv2.LINE_AA)


def run_menu() -> MenuChoice:
    detector    = HandDetector()
    shaka_start = None
    cap         = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
    cv2.namedWindow("MOTION MIX", cv2.WINDOW_AUTOSIZE)

    hold_start:  dict  = {}
    t0           = time.time()
    flash_until  = 0.0
    pending      = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        t      = time.time() - t0

        dark_overlay(frame, strength=0.50)

        (palm_pos, fist, lms), _ = detector.detect(frame, int(time.time() * 1000))
        if lms:
            detector.draw_skeleton(frame, lms, fw, fh, C_PLUM, C_MAGENTA)

        # shaka = quit, gotta hold it for SHAKA_HOLD seconds so it's not accidental
        shaka = bool(lms and HandDetector.is_shaka(lms))
        if shaka:
            if shaka_start is None:
                shaka_start = time.time()
            shaka_prog = min((time.time() - shaka_start) / SHAKA_HOLD, 1.0)
            if shaka_prog >= 1.0 and pending is None:
                pending, flash_until = MenuChoice.QUIT, time.time() + 0.22
        else:
            shaka_start = None
            shaka_prog  = 0.0

        bx0    = (fw - 3 * BTN_W - 2 * BTN_GAP) // 2
        by0    = fh // 2 - BTN_H // 2 + 30
        action = None

        for i, btn in enumerate(BUTTONS):
            bx, by = bx0 + i * (BTN_W + BTN_GAP), by0
            bid    = btn["id"]
            hover  = bool(
                palm_pos
                and bx <= palm_pos[0] <= bx + BTN_W
                and by <= palm_pos[1] <= by + BTN_H
            )
            progress = 0.0
            if hover and fist:
                hold_start.setdefault(bid, time.time())
                progress = min((time.time() - hold_start[bid]) / HOLD_SECS, 1.0)
                if progress >= 1.0 and pending is None:
                    pending, flash_until = bid, time.time() + 0.18
            else:
                hold_start.pop(bid, None)

            draw_button(frame, bx, by, BTN_W, BTN_H,
                        btn["label"], btn["icon"], hover, progress, t)

        draw_title(frame, "MOTION MIX", fw // 2, 74)
        draw_cursor(frame, palm_pos, fist, hold_start, HOLD_SECS)
        _draw_shaka_hint(frame, fw, fh, shaka_prog)

        # short white flash on activation = feels more responsive
        now = time.time()
        if now < flash_until:
            flash(frame, alpha=max(0.6 * (1 - (now - (flash_until - 0.18)) / 0.18), 0))
        if pending and now >= flash_until:
            action = pending

        cv2.imshow("MOTION MIX", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            action = MenuChoice.QUIT
        if action:
            cap.release()
            cv2.destroyAllWindows()
            detector.close()
            return action

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    return MenuChoice.QUIT
