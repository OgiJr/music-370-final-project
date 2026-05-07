import math
import os
import shutil
import time
from pathlib import Path

os.environ["GLOG_minloglevel"] = os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

from core.hand_detection import HandDetector
from core.gui import (
    C_AMBER, C_DARK, C_MAGENTA, C_PLUM, C_ROSE, C_VOID, C_WHITE,
    FONT, HOLD_SECS, WIN_H, WIN_W,
    dark_overlay, draw_cursor, draw_title, flash,
)
from views.demo_mix import run_mix

# four parents up because Python/v2/views/choose_sound.py
_ROOT_DIR   = Path(__file__).parent.parent.parent.parent
_MAX_DIR    = _ROOT_DIR / "Max"

SLOT_LABELS  = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
SLOT_NAMES   = ["One", "Two", "Three", "Four"]
BTN_W, BTN_H = 320, 170
BTN_GAP_X    = 50
BTN_GAP_Y    = 35
START_W      = 300
START_H      = 80
SHAKA_HOLD   = 0.8


def _draw_upload_arrow(frame: np.ndarray, cx: int, cy: int, size: int, col) -> None:
    hw = size // 2
    tip_y    = cy - hw
    shaft_ty = cy - hw // 3
    shaft_by = cy + hw // 2
    shaft_hw = max(hw // 5, 3)
    arr_hw   = hw // 2

    pts = np.array([
        [cx, tip_y],
        [cx - arr_hw, shaft_ty],
        [cx - shaft_hw, shaft_ty],
        [cx - shaft_hw, shaft_by],
        [cx + shaft_hw, shaft_by],
        [cx + shaft_hw, shaft_ty],
        [cx + arr_hw, shaft_ty],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], col)


def _draw_check(frame: np.ndarray, cx: int, cy: int, r: int, col) -> None:
    cv2.circle(frame, (cx, cy), r, col, 2, cv2.LINE_AA)
    p1 = (cx - r // 2, cy)
    p2 = (cx - r // 6, cy + r // 3)
    p3 = (cx + r // 2, cy - r // 3)
    cv2.line(frame, p1, p2, col, 2, cv2.LINE_AA)
    cv2.line(frame, p2, p3, col, 2, cv2.LINE_AA)


def _draw_slot_button(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str,
    loaded_path: "str | None",
    hover: bool,
    progress: float,
    t: float,
) -> None:
    roi    = frame[y:y + h, x:x + w]
    blurred = cv2.GaussianBlur(roi, (21, 21), 10)
    if loaded_path:
        # green-ish tint when a file is loaded so you can tell at a glance
        tint = np.full_like(blurred, (40, 80, 20))
    else:
        tint = np.full_like(blurred, C_PLUM if hover else C_VOID)
    frame[y:y + h, x:x + w] = cv2.addWeighted(blurred, 0.60, tint, 0.40, 0)

    border_col = (80, 200, 80) if loaded_path else (C_ROSE if hover else C_PLUM)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_col, 2, cv2.LINE_AA)

    if hover and not loaded_path:
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        fh_, fw_ = frame.shape[:2]
        for spread in (12, 5):
            rx1 = max(x - spread, 0);       ry1 = max(y - spread, 0)
            rx2 = min(x + w + spread, fw_); ry2 = min(y + h + spread, fh_)
            roi_b = frame[ry1:ry2, rx1:rx2]
            tmp   = roi_b.copy()
            cv2.rectangle(tmp,
                          (x - spread - rx1, y - spread - ry1),
                          (x + w + spread - rx1, y + h + spread - ry1),
                          C_MAGENTA, 1)
            cv2.addWeighted(roi_b, 1 - 0.07 * pulse, tmp, 0.07 * pulse, 0, roi_b)

    cx_ = x + w // 2

    (lw, _), _ = cv2.getTextSize(label, FONT, 0.52, 1)
    cv2.putText(frame, label, (cx_ - lw // 2, y + 28), FONT, 0.52,
                C_WHITE if hover else C_ROSE, 1, cv2.LINE_AA)

    if loaded_path:
        _draw_check(frame, cx_, y + h // 2 - 6, 18, (100, 230, 100))
        name = Path(loaded_path).name
        if len(name) > 26:
            name = name[:23] + "..."
        (nw, _), _ = cv2.getTextSize(name, FONT, 0.40, 1)
        cv2.putText(frame, name, (cx_ - nw // 2, y + h - 22), FONT, 0.40,
                    (140, 240, 140), 1, cv2.LINE_AA)
    else:
        icon_col = C_AMBER if hover else C_ROSE
        _draw_upload_arrow(frame, cx_, y + h // 2 - 6, 36, icon_col)
        sub = "fist-hold to browse"
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.38, 1)
        cv2.putText(frame, sub, (cx_ - sw // 2, y + h - 20), FONT, 0.38,
                    C_WHITE if hover else C_PLUM, 1, cv2.LINE_AA)

    if progress > 0:
        cv2.ellipse(frame, (cx_, y + h - 22), (15, 15),
                    -90, 0, int(360 * progress), C_AMBER, 3, cv2.LINE_AA)


def _draw_start_button(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    hover: bool,
    progress: float,
    t: float,
) -> None:
    roi     = frame[y:y + h, x:x + w]
    blurred = cv2.GaussianBlur(roi, (21, 21), 10)
    tint    = np.full_like(blurred, C_MAGENTA if hover else C_PLUM)
    frame[y:y + h, x:x + w] = cv2.addWeighted(blurred, 0.55, tint, 0.45, 0)

    cv2.rectangle(frame, (x, y), (x + w, y + h),
                  C_AMBER if hover else C_ROSE, 2, cv2.LINE_AA)

    if hover:
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        fh_, fw_ = frame.shape[:2]
        for spread in (14, 6):
            rx1 = max(x - spread, 0);       ry1 = max(y - spread, 0)
            rx2 = min(x + w + spread, fw_); ry2 = min(y + h + spread, fh_)
            roi_b = frame[ry1:ry2, rx1:rx2]
            tmp   = roi_b.copy()
            cv2.rectangle(tmp,
                          (x - spread - rx1, y - spread - ry1),
                          (x + w + spread - rx1, y + h + spread - ry1),
                          C_AMBER, 1)
            cv2.addWeighted(roi_b, 1 - 0.09 * pulse, tmp, 0.09 * pulse, 0, roi_b)

    label = "START MIX"
    (lw, _), _ = cv2.getTextSize(label, FONT, 0.75, 1)
    cx_ = x + w // 2
    cy_ = y + h // 2 + 8
    cv2.putText(frame, label, (cx_ - lw // 2, cy_), FONT, 0.75,
                C_AMBER if hover else C_WHITE, 1, cv2.LINE_AA)

    if progress > 0:
        cv2.ellipse(frame, (cx_, cy_ + 18), (18, 18),
                    -90, 0, int(360 * progress), C_AMBER, 3, cv2.LINE_AA)


def _open_file_dialog(slot_label: str) -> "str | None":
    # using applescript to get a native picker because tkinter looks like garbage on mac
    import subprocess
    script = (
        f'tell application "Finder" to activate\n'
        f'try\n'
        f'  set f to choose file with prompt "Select WAV for {slot_label}:"\n'
        f'  return POSIX path of f\n'
        f'on error\n'
        f'  return ""\n'
        f'end try'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=300,
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception:
        return None


def run_choose_sound() -> None:
    slots: list = [None, None, None, None]

    detector    = HandDetector()
    shaka_start = None
    cap         = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
    cv2.namedWindow("MOTION MIX", cv2.WINDOW_AUTOSIZE)

    hold_start:   dict  = {}
    t0            = time.time()
    flash_until   = 0.0
    pending_slot  = None
    pending_start = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        t      = time.time() - t0
        now    = time.time()

        dark_overlay(frame, strength=0.55)

        (palm_pos, fist, lms), _ = detector.detect(frame, int(time.time() * 1000))
        if lms:
            detector.draw_skeleton(frame, lms, fw, fh, C_PLUM, C_MAGENTA)

        shaka = bool(lms and HandDetector.is_shaka(lms))
        if shaka:
            if shaka_start is None:
                shaka_start = time.time()
            shaka_prog = min((time.time() - shaka_start) / SHAKA_HOLD, 1.0)
            if shaka_prog >= 1.0:
                break
        else:
            shaka_start = None
            shaka_prog  = 0.0

        all_loaded = all(s is not None for s in slots)

        grid_w = 2 * BTN_W + BTN_GAP_X
        grid_h = 2 * BTN_H + BTN_GAP_Y
        gx0    = (fw - grid_w) // 2
        # nudge grid up when start button shows so it's still vertically centered
        offset = (START_H + BTN_GAP_Y) // 2 if all_loaded else 0
        gy0    = (fh - grid_h) // 2 - offset + 30

        for i in range(4):
            col_i = i % 2
            row_i = i // 2
            bx    = gx0 + col_i * (BTN_W + BTN_GAP_X)
            by    = gy0 + row_i * (BTN_H + BTN_GAP_Y)

            hover = bool(
                palm_pos
                and bx <= palm_pos[0] <= bx + BTN_W
                and by <= palm_pos[1] <= by + BTN_H
            )
            progress = 0.0
            key      = f"slot_{i}"
            if hover and fist:
                hold_start.setdefault(key, time.time())
                progress = min((time.time() - hold_start[key]) / HOLD_SECS, 1.0)
                if (progress >= 1.0
                        and pending_slot is None
                        and not pending_start):
                    pending_slot = i
                    flash_until  = time.time() + 0.18
            else:
                hold_start.pop(key, None)

            _draw_slot_button(frame, bx, by, BTN_W, BTN_H,
                              SLOT_LABELS[i], slots[i], hover, progress, t)

        if all_loaded:
            sx = (fw - START_W) // 2
            sy = gy0 + grid_h + BTN_GAP_Y

            hover_s = bool(
                palm_pos
                and sx <= palm_pos[0] <= sx + START_W
                and sy <= palm_pos[1] <= sy + START_H
            )
            prog_s = 0.0
            if hover_s and fist:
                hold_start.setdefault("start", time.time())
                prog_s = min((time.time() - hold_start["start"]) / HOLD_SECS, 1.0)
                if (prog_s >= 1.0
                        and pending_slot is None
                        and not pending_start):
                    pending_start = True
                    flash_until   = time.time() + 0.22
            else:
                hold_start.pop("start", None)

            _draw_start_button(frame, sx, sy, START_W, START_H, hover_s, prog_s, t)

        draw_title(frame, "CHOOSE YOUR SOUND", fw // 2, 64)
        draw_cursor(frame, palm_pos, fist, hold_start, HOLD_SECS)

        hint_col = C_AMBER if shaka_prog > 0 else C_PLUM
        hint = "keys 1–4 also open picker  |  hold shaka to go back"
        (hw_, _), _ = cv2.getTextSize(hint, FONT, 0.38, 1)
        cv2.putText(frame, hint, ((fw - hw_) // 2, fh - 12),
                    FONT, 0.38, hint_col, 1, cv2.LINE_AA)

        if now < flash_until:
            elapsed = now - (flash_until - 0.18)
            flash(frame, alpha=max(0.6 * (1 - elapsed / 0.18), 0))

        cv2.imshow("MOTION MIX", frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):
            break

        # backup keyboard shortcuts in case the gestures are being annoying
        if k in (ord("1"), ord("2"), ord("3"), ord("4")):
            idx = k - ord("1")
            if pending_slot is None and not pending_start:
                pending_slot = idx
                flash_until  = time.time() + 0.05

        # have to defer opening the picker until after waitKey returns,
        # otherwise the file dialog blocks the camera and everything goes weird
        if pending_slot is not None and now >= flash_until:
            idx          = pending_slot
            pending_slot = None
            path = _open_file_dialog(SLOT_LABELS[idx])
            if path:
                slots[idx] = path
            hold_start.clear()

        if pending_start and now >= flash_until:
            pending_start = False
            _MAX_DIR.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(slots):
                shutil.copy2(src, _MAX_DIR / f"{i + 1}.wav")
            cap.release()
            cv2.destroyAllWindows()
            detector.close()
            run_mix(num_channels=4, names=SLOT_NAMES)
            return

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
