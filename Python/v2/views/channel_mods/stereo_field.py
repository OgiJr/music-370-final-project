import math
import collections

import cv2

from core.gui import (
    C_VOID, C_PLUM, C_ROSE, C_AMBER, C_WHITE, FONT,
)
from views.channel_mods._layout import TITLE_H, HINT_H, TAB_W

TRAIL_LEN = 40

_PAN_LABELS: dict[float, str] = {0.0: "L", 0.25: "L50", 0.5: "C", 0.75: "R50", 1.0: "R"}
_VOL_LABELS: dict[float, str] = {
    0.0: "-inf", 0.25: "-12 dB", 0.5: "-6 dB", 0.75: "-2 dB", 1.0: "0 dB"
}


def _bounds(fw: int, fh: int) -> tuple[int, int, int, int]:
    return TAB_W + 80, fw - 50, TITLE_H + 40, fh - HINT_H - 40


def draw(
    frame,
    fw: int,
    fh: int,
    chan_state: dict,
    r_palm,
    trail: collections.deque,
) -> None:
    gx1, gx2, gy1, gy2 = _bounds(fw, fh)

    # XY pad: hand position directly maps to pan + volume, no gestures needed
    if r_palm:
        pan    = max(0.0, min(1.0, (r_palm[0] - gx1) / max(gx2 - gx1, 1)))
        volume = max(0.0, min(1.0, 1 - (r_palm[1] - gy1) / max(gy2 - gy1, 1)))
        chan_state["pan"]    = pan
        chan_state["volume"] = volume
        trail.append((pan, volume))

    pan    = chan_state.get("pan",    0.50)
    volume = chan_state.get("volume", 0.80)

    dot_x = int(gx1 + pan    * (gx2 - gx1))
    dot_y = int(gy2 - volume * (gy2 - gy1))

    ov = frame.copy()
    cv2.rectangle(ov, (gx1, gy1), (gx2, gy2), C_VOID, -1)
    cv2.addWeighted(frame, 0.78, ov, 0.22, 0, frame)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # center line is brighter so you can see the C/0dB reference
        is_center = frac == 0.5
        col       = C_ROSE if is_center else C_PLUM
        thick     = 2      if is_center else 1

        vx = int(gx1 + frac * (gx2 - gx1))
        cv2.line(frame, (vx, gy1), (vx, gy2), col, thick, cv2.LINE_AA)

        hy = int(gy2 - frac * (gy2 - gy1))
        cv2.line(frame, (gx1, hy), (gx2, hy), col, thick, cv2.LINE_AA)

    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), C_PLUM, 2, cv2.LINE_AA)

    for frac, lbl in _PAN_LABELS.items():
        vx = int(gx1 + frac * (gx2 - gx1))
        (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.42, 1)
        cv2.putText(frame, lbl, (vx - tw // 2, gy2 + 22),
                    FONT, 0.42, C_PLUM, 1, cv2.LINE_AA)

    for frac, lbl in _VOL_LABELS.items():
        hy = int(gy2 - frac * (gy2 - gy1))
        (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.38, 1)
        cv2.putText(frame, lbl, (gx1 - tw - 8, hy + 5),
                    FONT, 0.38, C_PLUM, 1, cv2.LINE_AA)

    # motion trail: older points fade out so you can see where you've been
    pts = list(trail)
    for j in range(1, len(pts)):
        alpha = j / len(pts)
        sx = int(gx1 + pts[j][0]   * (gx2 - gx1))
        sy = int(gy2 - pts[j][1]   * (gy2 - gy1))
        ex = int(gx1 + pts[j - 1][0] * (gx2 - gx1))
        ey = int(gy2 - pts[j - 1][1] * (gy2 - gy1))
        col = tuple(int(c * alpha) for c in C_AMBER)
        cv2.line(frame, (ex, ey), (sx, sy), col, max(1, int(3 * alpha)), cv2.LINE_AA)

    cv2.line(frame, (gx1, dot_y), (gx2, dot_y), C_AMBER, 1, cv2.LINE_AA)
    cv2.line(frame, (dot_x, gy1), (dot_x, gy2), C_AMBER, 1, cv2.LINE_AA)

    fill = C_AMBER if r_palm else C_PLUM
    cv2.circle(frame, (dot_x, dot_y), 16, fill,   -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, dot_y), 22, C_WHITE,  2, cv2.LINE_AA)

    db_val  = 20 * math.log10(max(volume, 1e-4))
    vol_str = "-inf" if volume < 0.01 else f"{db_val:+.0f} dB"
    pv      = pan * 2 - 1
    pan_str = "C" if abs(pv) < 0.05 else (
        f"L {int(-pv * 100)}" if pv < 0 else f"R {int(pv * 100)}"
    )
    badge = f"  {vol_str}  |  {pan_str}  "
    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.52, 1)
    # nudge badge so it doesn't go off the right edge or top of the pad
    bx = min(dot_x + 26, gx2 - bw - 6)
    by = max(dot_y - 10, gy1 + bh + 4)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (bx - 4, by - bh - 4), (bx + bw + 4, by + 6), C_VOID, -1)
    cv2.addWeighted(frame, 0.55, ov2, 0.45, 0, frame)
    cv2.rectangle(frame, (bx - 4, by - bh - 4), (bx + bw + 4, by + 6),
                  C_AMBER, 1, cv2.LINE_AA)
    cv2.putText(frame, badge, (bx, by), FONT, 0.52, C_WHITE, 1, cv2.LINE_AA)
