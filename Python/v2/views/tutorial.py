import math
import os
import time

os.environ["GLOG_minloglevel"] = os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

from core.hand_detection import HandDetector
from core.gui import (
    C_AMBER, C_DARK, C_MAGENTA, C_PLUM, C_ROSE, C_VOID, C_WHITE,
    FONT, WIN_H, WIN_W,
    dark_overlay, draw_title,
)

TITLE_H    = 88
HINT_H     = 44
SHAKA_HOLD = 0.8
NUM_PAGES  = 5

PAGE_ACCENT = [C_AMBER, C_MAGENTA, C_AMBER, C_ROSE, C_PLUM]
PAGE_TITLES = [
    "GETTING STARTED",
    "THE 5 GESTURES",
    "NAVIGATING MENUS",
    "MIX & MASTER",
    "CHANNEL EFFECTS",
]


def _card(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    accent,
    title: str,
    lines: list,
    line_scale: float = 0.43,
) -> int:
    # blur background + tint = glassmorphic card thingy
    roi     = frame[y:y + h, x:x + w]
    blurred = cv2.GaussianBlur(roi, (21, 21), 10)
    tint    = np.full_like(blurred, C_VOID)
    frame[y:y + h, x:x + w] = cv2.addWeighted(blurred, 0.55, tint, 0.45, 0)

    HH = 36
    hdr  = frame[y:y + HH, x:x + w].astype(np.float32)
    col  = np.full_like(hdr, accent, dtype=np.float32)
    frame[y:y + HH, x:x + w] = np.clip(hdr * 0.45 + col * 0.55, 0, 255).astype(np.uint8)

    cv2.rectangle(frame, (x, y), (x + w, y + h), accent, 2, cv2.LINE_AA)

    (tw, _), _ = cv2.getTextSize(title, FONT, 0.60, 1)
    cv2.putText(frame, title, (x + (w - tw) // 2, y + 25),
                FONT, 0.60, C_WHITE, 1, cv2.LINE_AA)

    ly    = y + HH + 20
    lh    = int(line_scale * 52) + 4
    for line in lines:
        # cheap markup: ## = subheader, >> = highlighted bullet, "  " = white
        if line == "":
            ly += lh // 2
            continue
        if line.startswith("##"):
            text = line[2:].strip()
            (lw, _), _ = cv2.getTextSize(text, FONT, 0.50, 1)
            cv2.putText(frame, text, (x + (w - lw) // 2, ly),
                        FONT, 0.50, C_AMBER, 1, cv2.LINE_AA)
        elif line.startswith(">>"):
            text = line[2:].strip()
            cv2.putText(frame, text, (x + 16, ly),
                        FONT, line_scale, C_AMBER, 1, cv2.LINE_AA)
        else:
            col_ = C_WHITE if line.startswith("  ") else C_ROSE
            cv2.putText(frame, line.strip(), (x + 16, ly),
                        FONT, line_scale, col_, 1, cv2.LINE_AA)
        ly += lh
    return ly


def _ico_palm(frame: np.ndarray, cx: int, cy: int, r: int, col) -> None:
    pr = r // 3
    cv2.circle(frame, (cx, cy + 8), pr, col, 2, cv2.LINE_AA)
    for ang in (-65, -32, 0, 32, 65):
        rad  = math.radians(ang - 90)
        x1   = cx + int(pr * math.cos(rad))
        y1   = (cy + 8) + int(pr * math.sin(rad))
        x2   = cx + int(r  * math.cos(rad))
        y2   = (cy + 8) + int(r  * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        cv2.circle(frame, (x2, y2), 5, col, -1, cv2.LINE_AA)


def _ico_fist(frame: np.ndarray, cx: int, cy: int, r: int, col) -> None:
    hw, hh = int(r * 0.82), int(r * 0.62)
    pts = np.array([
        [cx - hw, cy - hh + 12],
        [cx - hw + 8, cy - hh],
        [cx + hw - 8, cy - hh],
        [cx + hw, cy - hh + 12],
        [cx + hw, cy + hh],
        [cx - hw, cy + hh],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], col)
    cv2.polylines(frame, [pts], True, C_DARK, 2, cv2.LINE_AA)
    for i in range(1, 4):
        kx = cx - hw + i * (2 * hw // 4)
        cv2.line(frame, (kx, cy - hh), (kx, cy - hh + 14), C_DARK, 2, cv2.LINE_AA)


def _ico_pinch(frame: np.ndarray, cx: int, cy: int, r: int, col) -> None:
    tr = r // 5
    off = r // 3
    ix, iy = cx - off, cy - r // 4
    cv2.line(frame, (ix, iy - r // 2), (ix, iy), col, 3, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), tr, col, 2, cv2.LINE_AA)
    tx, ty = cx + off // 2, cy + r // 4
    cv2.line(frame, (tx + r // 2, ty + r // 3), (tx, ty), col, 3, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), tr, col, 2, cv2.LINE_AA)
    mx = (ix + tx) // 2
    my = (iy + ty) // 2
    cv2.line(frame, (ix, iy), (tx, ty), col, 2, cv2.LINE_AA)
    cv2.circle(frame, (mx, my), 8, C_AMBER, -1, cv2.LINE_AA)


def _ico_twist(frame: np.ndarray, cx: int, cy: int, r: int, col) -> None:
    tr = r // 5
    cv2.circle(frame, (cx - r // 3, cy + r // 5), tr, col, 2, cv2.LINE_AA)
    cv2.line(frame, (cx - r // 3, cy + r // 5), (cx - r // 3, cy - r // 2), col, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx + r // 3, cy + r // 5), tr, col, 2, cv2.LINE_AA)
    cv2.line(frame, (cx + r // 3, cy + r // 5), (cx + r // 3, cy - r // 2), col, 3, cv2.LINE_AA)
    cv2.line(frame, (cx - r // 3, cy + r // 5), (cx + r // 3, cy + r // 5), col, 2, cv2.LINE_AA)
    arc_r = int(r * 0.68)
    cv2.ellipse(frame, (cx, cy + r // 3), (arc_r, arc_r // 2),
                0, 200, 340, C_AMBER, 2, cv2.LINE_AA)
    arad = math.radians(340)
    ax   = cx + int(arc_r * math.cos(arad))
    ay   = (cy + r // 3) + int((arc_r // 2) * math.sin(arad))
    cv2.arrowedLine(frame, (ax - 10, ay - 6), (ax, ay),
                    C_AMBER, 2, cv2.LINE_AA, tipLength=0.6)


def _ico_shaka(frame: np.ndarray, cx: int, cy: int, r: int, col) -> None:
    pr = r // 4
    cv2.circle(frame, (cx, cy), pr, col, 2, cv2.LINE_AA)
    cv2.line(frame, (cx - pr, cy), (cx - r, cy - r // 5), col, 4, cv2.LINE_AA)
    cv2.circle(frame, (cx - r, cy - r // 5), 6, col, -1, cv2.LINE_AA)
    cv2.line(frame, (cx + pr, cy), (cx + r, cy - r // 5), col, 4, cv2.LINE_AA)
    cv2.circle(frame, (cx + r, cy - r // 5), 6, col, -1, cv2.LINE_AA)
    for i in (-1, 0, 1):
        bx = cx + i * (r // 3)
        cv2.ellipse(frame, (bx, cy - pr), (r // 6, r // 7),
                    0, 180, 360, col, 2, cv2.LINE_AA)


def _page0(frame: np.ndarray, fw: int, fh: int, t: float = 0.0) -> None:
    accent  = C_AMBER
    mx      = 80
    gap     = 26
    cw_narrow = 520
    cw_left   = fw - 2 * mx - gap - cw_narrow
    cw_right  = cw_narrow

    NAV_BANNER_H = 62
    ch  = 560
    cy0 = TITLE_H + (fh - TITLE_H - HINT_H - ch - NAV_BANNER_H - 14) // 2 + 6

    _card(frame, mx, cy0, cw_left, ch, C_AMBER, "BEFORE YOU START", [
        ">> Step 1 - Run the app",
        "  python main.py  (inside the v2 venv)",
        "",
        ">> Step 2 - Pick a mode",
        "  Choose  Try out Demo  or  Choose Your Sound",
        "",
        ">> Step 3 - Load your WAV files",
        "  (Choose Your Sound mode only)",
        "  Fist-hold each of the 4 slots to browse.",
        "  Then fist-hold START MIX to launch.",
        "",
        ">> Step 4 - Open Max / MSP",
        "  Open  motion-mix.maxpat  from the Max/ folder.",
        "",
        ">> Step 5 - Check sample rate  !!!",
        "  Options > Audio Status  must show  48000 Hz.",
        "  Wrong rate = pitch and speed problems.",
        "",
        ">> Step 6 - Turn on audio",
        "  Click the speaker icon or press Cmd + /",
        "  to start DSP.  Sound now follows gestures.",
    ], line_scale=0.42)

    _card(frame, mx + cw_left + gap, cy0, cw_right, ch, C_PLUM, "HOW IT CONNECTS", [
        "  This Python app tracks your",
        "  hands via your webcam and",
        "  sends control data over UDP",
        "  to Max / MSP on port 9000.",
        "",
        "  Max receives those messages",
        "  and applies them in real time",
        "  to the audio engine:",
        "",
        ">> Volume  - fader level per channel",
        ">> Pan     - stereo position",
        ">> Reverb  - room size & decay",
        ">> EQ      - filter type, freq & gain",
        ">> Stereo  - width & rotation",
        "",
        "  Both apps must be running",
        "  at the same time on the",
        "  same machine for gestures",
        "  to control the audio.",
        "",
        "  Press NEXT >> to read the",
        "  gesture guide.",
    ], line_scale=0.42)

    bh   = 62
    by   = cy0 + ch + 14
    bx   = mx
    bw   = fw - 2 * mx
    pulse = 0.55 + 0.45 * math.sin(t * 3.5)

    # the navigation banner pulses so people actually notice the arrow keys exist
    roi  = frame[by:by + bh, bx:bx + bw]
    fill = np.full_like(roi, C_AMBER)
    frame[by:by + bh, bx:bx + bw] = np.clip(
        roi.astype(np.float32) * (1 - pulse * 0.75) + fill.astype(np.float32) * (pulse * 0.75),
        0, 255
    ).astype(np.uint8)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), C_AMBER, 3, cv2.LINE_AA)

    tri_pts = np.array([[bx + 28, by + bh // 2],
                         [bx + 54, by + 10],
                         [bx + 54, by + bh - 10]], np.int32)
    cv2.fillPoly(frame, [tri_pts], C_DARK)

    tri_pts2 = np.array([[bx + bw - 28, by + bh // 2],
                          [bx + bw - 54, by + 10],
                          [bx + bw - 54, by + bh - 10]], np.int32)
    cv2.fillPoly(frame, [tri_pts2], C_DARK)

    msg = "USE THE  <<  >>  ARROW KEYS  TO  NAVIGATE  THIS  TUTORIAL"
    (mw, _), _ = cv2.getTextSize(msg, FONT, 0.80, 2)
    cv2.putText(frame, msg, (fw // 2 - mw // 2, by + bh // 2 + 10),
                FONT, 0.80, C_DARK, 2, cv2.LINE_AA)


def _page1(frame: np.ndarray, fw: int, fh: int) -> None:
    n       = 5
    mx      = 55
    gap     = 14
    avail_w = fw - 2 * mx
    cw      = (avail_w - gap * (n - 1)) // n
    ch      = 430
    cy0     = TITLE_H + (fh - TITLE_H - HINT_H - ch) // 2 + 10

    gestures = [
        ("OPEN PALM",  _ico_palm,  C_MAGENTA, [
            "  Hover to aim the cursor",
            "  over any button or slot.",
        ]),
        ("FIST + HOLD", _ico_fist, C_PLUM, [
            "  Curl all fingers into a",
            "  fist and hold ~0.9 s to",
            "  activate the hovered item.",
        ]),
        ("PINCH",      _ico_pinch, C_ROSE, [
            "  Bring thumb + index",
            "  together to grab a",
            "  volume fader.",
        ]),
        ("TWIST",      _ico_twist, C_AMBER, [
            "  Bring thumb + middle",
            "  together, then rotate",
            "  your wrist for pan.",
        ]),
        ("SHAKA",      _ico_shaka, C_PLUM, [
            "  Thumb + pinky out,",
            "  three middle fingers",
            "  curled. Hold 0.8 s",
            "  to go back / quit.",
        ]),
    ]

    for i, (name, icon_fn, accent, desc) in enumerate(gestures):
        cx  = mx + i * (cw + gap)
        roi = frame[cy0:cy0 + ch, cx:cx + cw]
        blr = cv2.GaussianBlur(roi, (21, 21), 10)
        tnt = np.full_like(blr, C_VOID)
        frame[cy0:cy0 + ch, cx:cx + cw] = cv2.addWeighted(blr, 0.55, tnt, 0.45, 0)
        HH = 36
        hdr = frame[cy0:cy0 + HH, cx:cx + cw].astype(np.float32)
        hc  = np.full_like(hdr, accent, dtype=np.float32)
        frame[cy0:cy0 + HH, cx:cx + cw] = np.clip(hdr * 0.45 + hc * 0.55, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (cx, cy0), (cx + cw, cy0 + ch), accent, 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(name, FONT, 0.57, 1)
        cv2.putText(frame, name, (cx + (cw - tw) // 2, cy0 + 24),
                    FONT, 0.57, C_WHITE, 1, cv2.LINE_AA)

        # ended up just drawing a giant number instead of icons, looked cleaner
        num  = str(i + 1)
        nscale = 3.8
        nthick = 6
        (nw, nh), _ = cv2.getTextSize(num, FONT, nscale, nthick)
        icx = cx + cw // 2
        icy = cy0 + HH + int((ch - HH) * 0.50) - 100
        cv2.putText(frame, num, (icx - nw // 2, icy + nh // 2),
                    FONT, nscale, accent, nthick, cv2.LINE_AA)

        sep_y = cy0 + HH + int((ch - HH) * 0.62)
        cv2.line(frame, (cx + 12, sep_y), (cx + cw - 12, sep_y),
                 C_PLUM, 1, cv2.LINE_AA)

        dy = sep_y + 20
        for line in desc:
            (lw, _), _ = cv2.getTextSize(line.strip(), FONT, 0.42, 1)
            cv2.putText(frame, line.strip(), (cx + (cw - lw) // 2, dy),
                        FONT, 0.42, C_WHITE, 1, cv2.LINE_AA)
            dy += 24


def _page2(frame: np.ndarray, fw: int, fh: int) -> None:
    accent = C_AMBER
    mx     = 80
    gap    = 30
    cw     = (fw - 2 * mx - gap) // 2
    ch     = 520
    cy0    = TITLE_H + (fh - TITLE_H - HINT_H - ch) // 2 + 6

    _card(frame, mx, cy0, cw, ch, accent, "MAIN MENU", [
        ">> How to select a button:",
        "",
        "- Point your open right hand toward",
        "  the screen to move the cursor.",
        "",
        "- Hover your palm over a button,",
        "  it will glow when highlighted.",
        "",
        "- Make a FIST and hold it (~0.9 s).",
        "  A gold progress arc fills up,",
        "  then the option activates.",
        "",
        ">> To quit the application:",
        "",
        "- Show a SHAKA and hold for 0.8 s",
        "  from the main menu screen.",
    ], line_scale=0.42)

    rx = mx + cw + gap
    _card(frame, rx, cy0, cw, ch, C_ROSE, "CHOOSE YOUR SOUND", [
        ">> Loading your WAV files:",
        "",
        "- Four slots appear in a 2x2 grid,",
        "  one per mixer channel.",
        "",
        "- Hover + FIST-HOLD any slot to",
        "  open a native file browser.",
        "",
        "- Alternatively press keys 1 - 4",
        "  to open the picker for that slot.",
        "",
        "- A green tick shows when a slot",
        "  has a file loaded.",
        "",
        ">> Once all four slots are filled:",
        "",
        "- A START MIX button appears below.",
        "  Fist-hold it to copy your files",
        "  and launch the mixer.",
    ], line_scale=0.42)


def _page3(frame: np.ndarray, fw: int, fh: int) -> None:
    accent = C_ROSE
    mx     = 55
    gap    = 22
    n      = 3
    avail  = fw - 2 * mx - gap * (n - 1)
    cw     = avail // n
    ch     = 520
    cy0    = TITLE_H + (fh - TITLE_H - HINT_H - ch) // 2 + 6

    _card(frame, mx, cy0, cw, ch, C_ROSE, "VOLUME  (R HAND)", [
        "## PINCH  =  thumb + index finger",
        "",
        "- Move your right hand until",
        "  your cursor is over the",
        "  channel you want to adjust.",
        "",
        "- Pinch thumb to index to",
        "  grab that channel's fader.",
        "",
        "- Drag your hand UP to raise",
        "  the volume.",
        "- Drag your hand DOWN to",
        "  lower the volume.",
        "",
        "- Separate fingers to release.",
        "  The fader stays in place.",
        "",
        "  Tip: The dB value updates",
        "  live next to the fader.",
    ], line_scale=0.41)

    _card(frame, mx + cw + gap, cy0, cw, ch, C_AMBER, "PAN  (R HAND)", [
        "## TWIST  =  thumb + middle finger",
        "",
        "- Hover your right hand over",
        "  the channel you want to pan.",
        "",
        "- Pinch your thumb to your",
        "  MIDDLE finger to grab the",
        "  pan knob.",
        "",
        "- Rotate your wrist:",
        "  - Clockwise  =  pan RIGHT",
        "  - Counter-CW =  pan LEFT",
        "",
        "- Separate fingers to release.",
        "",
        "  The knob label shows",
        "  L##, C (centre), or R##.",
        "",
        "  Tip: Twist and Pinch cannot",
        "  be active at the same time.",
    ], line_scale=0.41)

    _card(frame, mx + 2 * (cw + gap), cy0, cw, ch, C_PLUM, "OPEN EFFECTS  (L HAND)", [
        "## Hold N fingers on LEFT hand",
        "",
        "- Raise 1 finger on your left",
        "  hand and hold for 1.5 s",
        "  to open Channel 1 effects.",
        "",
        "- 2 fingers  =  Channel 2",
        "- 3 fingers  =  Channel 3",
        "- 4 fingers  =  Channel 4",
        "",
        "  A gold progress arc spins",
        "  above the channel name",
        "  while you hold.",
        "",
        ">> Returning to the mixer:",
        "",
        "- Hold SHAKA (right hand)",
        "  for 0.8 s inside any",
        "  effects screen.",
    ], line_scale=0.41)


def _page4(frame: np.ndarray, fw: int, fh: int) -> None:
    accent = C_PLUM
    mx     = 55
    gap    = 22
    n      = 3
    avail  = fw - 2 * mx - gap * (n - 1)
    cw     = avail // n
    ch     = 520
    cy0    = TITLE_H + (fh - TITLE_H - HINT_H - ch) // 2 + 6

    badge_y = cy0 - 34
    tab_labels = [
        ("L-HAND  1 FINGER", "TAB 1 : STEREO FIELD"),
        ("L-HAND  2 FINGERS", "TAB 2 : REVERB"),
        ("L-HAND  3 FINGERS", "TAB 3 : EQ"),
    ]
    tab_cols = [C_ROSE, C_AMBER, C_PLUM]
    for i, ((badge, tab_name), tcol) in enumerate(zip(tab_labels, tab_cols)):
        bx = mx + i * (cw + gap)
        roi = frame[badge_y:badge_y + 28, bx:bx + cw]
        tnt = np.full_like(roi, tcol)
        frame[badge_y:badge_y + 28, bx:bx + cw] = cv2.addWeighted(roi, 0.5, tnt, 0.5, 0)
        cv2.rectangle(frame, (bx, badge_y), (bx + cw, badge_y + 28), tcol, 1, cv2.LINE_AA)
        (bw, _), _ = cv2.getTextSize(badge, FONT, 0.38, 1)
        cv2.putText(frame, badge, (bx + (cw - bw) // 2, badge_y + 18),
                    FONT, 0.38, C_WHITE, 1, cv2.LINE_AA)

    _card(frame, mx, cy0, cw, ch, C_ROSE, "STEREO FIELD", [
        "  Right hand is a free XY pad",
        "  covering the whole screen.",
        "",
        ">> X-axis (left/right):",
        "  Move right hand left or right",
        "  to sweep the PAN position.",
        "",
        ">> Y-axis (up/down):",
        "  Move right hand up to raise",
        "  volume, down to lower it.",
        "",
        "  Changes update in real time",
        "  and are sent to Max/MSP.",
        "",
        "  A trail shows where your",
        "  hand has been moving.",
        "",
        "  Switch tabs: L-hand 2 = Reverb",
        "               L-hand 3 = EQ",
    ], line_scale=0.41)

    _card(frame, mx + cw + gap, cy0, cw, ch, C_AMBER, "REVERB", [
        ">> Toggle reverb ON / OFF:",
        "  Hold 1 finger (R hand) = OFF",
        "  Hold 2+ fingers        = ON",
        "",
        ">> Two vertical faders:",
        "  Pinch (thumb + index) to",
        "  grab and drag a slider.",
        "",
        "  Left fader  = Room Size",
        "    (larger = bigger space)",
        "",
        "  Right fader = Decay",
        "    (longer = slower fade)",
        "",
        "  Separate fingers to release.",
        "",
        "  The reverb status badge",
        "  shows ON or OFF at the",
        "  top of the screen.",
    ], line_scale=0.41)

    _card(frame, mx + 2 * (cw + gap), cy0, cw, ch, C_PLUM, "EQ", [
        ">> Select filter type:",
        "  Hold 1-5 fingers (R hand):",
        "  1 = Low-pass   2 = High-pass",
        "  3 = Band-pass  4 = Notch",
        "  5 = Peaking",
        "",
        ">> Adjust frequency & gain:",
        "  PINCH (thumb + index) and",
        "  move right hand:",
        "  X-axis = cutoff frequency",
        "  Y-axis = gain (+ / -)",
        "",
        ">> Adjust resonance (Q):",
        "  Make a FIST with left hand",
        "  and move it up / down.",
        "",
        "  The EQ curve graph updates",
        "  live as you move.",
    ], line_scale=0.41)


def _draw_arrows(
    frame: np.ndarray, fw: int, fh: int, page: int, t: float
) -> None:
    cy    = fh // 2
    pulse = int(180 + 60 * math.sin(t * 2.5))
    col   = (pulse, pulse, pulse)

    if page > 0:
        pts = np.array([[28, cy], [52, cy - 22], [52, cy + 22]], np.int32)
        cv2.fillPoly(frame, [pts], col)

    if page < NUM_PAGES - 1:
        pts = np.array([[fw - 28, cy], [fw - 52, cy - 22], [fw - 52, cy + 22]], np.int32)
        cv2.fillPoly(frame, [pts], col)


def _draw_dots(frame: np.ndarray, fw: int, fh: int, page: int) -> None:
    r   = 6
    gap = 22
    tx  = fw // 2 - (NUM_PAGES * gap) // 2 + gap // 2
    ty  = fh - HINT_H - 18

    for i in range(NUM_PAGES):
        col  = C_AMBER if i == page else C_PLUM
        fill = -1 if i == page else 2
        cv2.circle(frame, (tx + i * gap, ty), r, col, fill, cv2.LINE_AA)


def _draw_hint(
    frame: np.ndarray, fw: int, fh: int, page: int, shaka_prog: float
) -> None:
    nav = []
    if page > 0:
        nav.append("<< PREV")
    nav.append(f"page {page + 1} / {NUM_PAGES}")
    if page < NUM_PAGES - 1:
        nav.append("NEXT >>")

    hint  = "   |   ".join(nav) + "   |   hold shaka to exit"
    col   = C_AMBER if shaka_prog > 0 else C_PLUM
    (hw, th), _ = cv2.getTextSize(hint, FONT, 0.40, 1)
    hx = (fw - hw) // 2
    hy = fh - 12
    cv2.putText(frame, hint, (hx, hy), FONT, 0.40, col, 1, cv2.LINE_AA)
    if shaka_prog > 0:
        cv2.ellipse(frame, (hx + hw + 16, hy - th // 2), (8, 8),
                    -90, 0, int(360 * shaka_prog), C_AMBER, 2, cv2.LINE_AA)


_PAGE_FNS = [_page0, _page1, _page2, _page3, _page4]


def run_tutorial() -> None:
    detector    = HandDetector()
    shaka_start = None
    cap         = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
    cv2.namedWindow("MOTION MIX", cv2.WINDOW_AUTOSIZE)

    page = 0
    t0   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        t      = time.time() - t0

        dark_overlay(frame, strength=0.62)

        (_, _, r_lms), _ = detector.detect(frame, int(time.time() * 1000))

        shaka = bool(r_lms and HandDetector.is_shaka(r_lms))
        if shaka:
            if shaka_start is None:
                shaka_start = time.time()
            shaka_prog = min((time.time() - shaka_start) / SHAKA_HOLD, 1.0)
            if shaka_prog >= 1.0:
                break
        else:
            shaka_start = None
            shaka_prog  = 0.0

        # only page 0 needs the timestamp because of the pulsing banner
        if page == 0:
            _PAGE_FNS[page](frame, fw, fh, t)
        else:
            _PAGE_FNS[page](frame, fw, fh)

        draw_title(frame, PAGE_TITLES[page], fw // 2, 60,
                   scale=1.5)
        _draw_arrows(frame, fw, fh, page, t)
        _draw_dots(frame, fw, fh, page)
        _draw_hint(frame, fw, fh, page, shaka_prog)

        cv2.line(frame, (80, TITLE_H - 4), (fw - 80, TITLE_H - 4),
                 PAGE_ACCENT[page], 1, cv2.LINE_AA)

        cv2.imshow("MOTION MIX", frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord("q"), 27):
            break
        # arrow keys send different codes on linux vs mac, so check both
        if k == 81 or k == 2:
            page = max(0, page - 1)
        if k == 83 or k == 3:
            page = min(NUM_PAGES - 1, page + 1)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
