import os
import math
import time as _time

import cv2
import numpy as np

# colors are in BGR because opencv is weird like that
C_VOID    = ( 80,   3,  48)
C_PLUM    = (127,  22, 148)
C_ROSE    = (121,  52, 233)
C_AMBER   = ( 83, 172, 249)
C_MAGENTA = (151,  46, 246)
C_WHITE   = (255, 255, 255)
C_DARK    = (  8,   8,  12)

FONT      = cv2.FONT_HERSHEY_DUPLEX
HOLD_SECS = 0.9
WIN_W, WIN_H = 1920, 1080

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

ICONS: dict[str, np.ndarray] = {}
for _name, _file in [("play", "play.png"), ("note", "music.png"), ("info", "info.png")]:
    _img = cv2.imread(os.path.join(_ASSETS_DIR, _file), cv2.IMREAD_UNCHANGED)
    if _img is not None:
        ICONS[_name] = _img


def time_now() -> float:
    return _time.time()


def dark_overlay(frame: np.ndarray, strength: float = 0.45) -> None:
    cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - strength, beta=0)


# resizing icons every frame killed performance, so cache them
_ICON_CACHE: dict[tuple, np.ndarray] = {}

def draw_icon(
    frame: np.ndarray, cx: int, cy: int, name: str, size: int, hover: bool
) -> None:
    raw = ICONS.get(name)
    if raw is None:
        return
    cache_key = (name, size)
    if cache_key not in _ICON_CACHE:
        _ICON_CACHE[cache_key] = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
    icon = _ICON_CACHE[cache_key]
    a    = (
        icon[:, :, 3:4]
        if icon.shape[2] == 4
        else cv2.cvtColor(icon[:, :, :3], cv2.COLOR_BGR2GRAY)[:, :, None]
    ).astype(np.float32) / 255.0
    flat = np.full((size, size, 3), C_AMBER if hover else C_ROSE, dtype=np.float32)
    fh, fw = frame.shape[:2]
    x,  y  = cx - size // 2, cy - size // 2
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + size, fw), min(y + size, fh)
    if x2 <= x1 or y2 <= y1:
        return
    sx, sy = x1 - x, y1 - y
    h_, w_ = y2 - y1, x2 - x1
    roi    = frame[y1:y2, x1:x2].astype(np.float32)
    a_r    = a[sy:sy + h_, sx:sx + w_]
    flat_r = flat[sy:sy + h_, sx:sx + w_]
    frame[y1:y2, x1:x2] = np.clip(roi * (1 - a_r) + flat_r * a_r, 0, 255).astype(np.uint8)


def draw_button(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str, icon: str,
    hover: bool, progress: float, t: float,
) -> None:
    # blur + tint = fake glassmorphism
    blurred = cv2.GaussianBlur(frame[y:y + h, x:x + w], (21, 21), 10)
    tint    = np.full_like(blurred, C_PLUM if hover else C_VOID)
    frame[y:y + h, x:x + w] = cv2.addWeighted(blurred, 0.62, tint, 0.38, 0)

    if hover:
        pulse    = 0.5 + 0.5 * math.sin(t * 5)
        fh_, fw_ = frame.shape[:2]
        for spread in (12, 5):
            # only blend the border strip, NOT the whole frame... was super slow before
            rx1 = max(x - spread, 0);     ry1 = max(y - spread, 0)
            rx2 = min(x + w + spread, fw_); ry2 = min(y + h + spread, fh_)
            roi = frame[ry1:ry2, rx1:rx2]
            tmp = roi.copy()
            cv2.rectangle(tmp,
                          (x - spread - rx1, y - spread - ry1),
                          (x + w + spread - rx1, y + h + spread - ry1),
                          C_MAGENTA, 1)
            cv2.addWeighted(roi, 1 - 0.07 * pulse, tmp, 0.07 * pulse, 0, roi)

    cv2.rectangle(frame, (x, y), (x + w, y + h),
                  C_ROSE if hover else C_PLUM, 2, cv2.LINE_AA)
    draw_icon(frame, x + w // 2, y + int(h * 0.38), icon, 68, hover)

    (tw, _), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    cv2.putText(frame, label, (x + (w - tw) // 2, y + h - 20),
                FONT, 0.55, C_WHITE if hover else C_ROSE, 1, cv2.LINE_AA)

    if progress > 0:
        cv2.ellipse(frame, (x + w // 2, y + h - 22), (15, 15),
                    -90, 0, int(360 * progress), C_AMBER, 3, cv2.LINE_AA)


def draw_title(
    frame: np.ndarray, text: str, cx: int, cy: int, scale: float = 1.8
) -> None:
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    tx = cx - tw // 2
    # double draw = fake drop shadow
    cv2.putText(frame, text, (tx, cy), FONT, scale, C_DARK,  8, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, cy), FONT, scale, C_WHITE, 2, cv2.LINE_AA)


def draw_cursor(
    frame: np.ndarray,
    palm_pos,
    fist: bool,
    hold_start: dict,
    hold_secs: float,
) -> None:
    if palm_pos is None:
        return
    px, py = palm_pos
    col    = C_MAGENTA if fist else C_AMBER
    cv2.circle(frame, (px, py), 20, col, 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py),  4, col, -1, cv2.LINE_AA)
    if fist and hold_start:
        prog = max(
            min((time_now() - ts) / hold_secs, 1.0) for ts in hold_start.values()
        )
        cv2.ellipse(frame, (px, py), (30, 30),
                    -90, 0, int(360 * prog), C_AMBER, 3, cv2.LINE_AA)


def flash(frame: np.ndarray, alpha: float = 0.6) -> None:
    cv2.addWeighted(frame, 1 - alpha,
                    np.full_like(frame, 255), alpha, 0, frame)
