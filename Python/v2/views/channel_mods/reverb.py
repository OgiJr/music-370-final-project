import time
from dataclasses import dataclass, field

import cv2

from core.gui import C_VOID, C_PLUM, C_ROSE, C_AMBER, C_MAGENTA, C_WHITE, FONT
from core.hand_detection import HandDetector
from views.channel_mods._layout import TITLE_H, HINT_H, TAB_W

# these are the actual ranges Max uses, but in the UI we just show 0-100%
SIZE_MIN,  SIZE_MAX  = 10,   100
DECAY_MIN, DECAY_MAX = 0.80, 0.98

LOCK_HOLD = 0.40

_PAD_TOP = 100
_PAD_BOT = 50

_SLIDER_KEYS = ["room_size", "decay"]

PILL_LABELS = ["OFF", "ON"]


@dataclass
class State:
    grab:       int          = -1
    was_fist:   bool         = False
    reverb_on:  bool         = False
    cand_count: int          = 0
    cand_start: float | None = None


def _layout(fw: int, fh: int) -> tuple:
    content_x1 = TAB_W
    content_w  = fw - TAB_W
    top        = TITLE_H + _PAD_TOP
    bot        = fh - HINT_H - _PAD_BOT
    cxs        = [
        content_x1 + content_w * 1 // 3,
        content_x1 + content_w * 2 // 3,
    ]
    return top, bot, cxs


def _v_slider(
    frame,
    cx: int,
    top: int,
    bot: int,
    val: float,
    label: str,
    lbl_bot: str,
    lbl_top: str,
    grabbed: bool,
) -> None:
    thumb_y = int(bot - val * (bot - top))

    cv2.line(frame, (cx, top), (cx, bot), C_PLUM, 4, cv2.LINE_AA)
    cv2.line(frame, (cx, thumb_y), (cx, bot),
             C_AMBER if grabbed else C_ROSE, 4, cv2.LINE_AA)

    fill   = C_MAGENTA if grabbed else C_PLUM
    border = C_AMBER   if grabbed else C_WHITE
    cv2.rectangle(frame, (cx - 20, thumb_y - 10), (cx + 20, thumb_y + 10), fill, -1)
    cv2.rectangle(frame, (cx - 20, thumb_y - 10), (cx + 20, thumb_y + 10),
                  border, 2 if grabbed else 1, cv2.LINE_AA)

    (lw, _), _ = cv2.getTextSize(label, FONT, 0.60, 1)
    cv2.putText(frame, label, (cx - lw // 2, top - 16),
                FONT, 0.60, C_WHITE, 1, cv2.LINE_AA)

    (bw, _), _ = cv2.getTextSize(lbl_bot, FONT, 0.40, 1)
    cv2.putText(frame, lbl_bot, (cx - bw // 2, bot + 22),
                FONT, 0.40, C_PLUM, 1, cv2.LINE_AA)

    (tw, _), _ = cv2.getTextSize(lbl_top, FONT, 0.40, 1)
    cv2.putText(frame, lbl_top, (cx - tw // 2, top - 38),
                FONT, 0.40, C_PLUM, 1, cv2.LINE_AA)


def update(
    data:   dict,
    r_fist: bool,
    r_palm,
    r_lms,
    fw:     int,
    fh:     int,
    state:  State,
) -> float:
    top, bot, cxs = _layout(fw, fh)
    hold_prog = 0.0

    pinching = bool(r_lms and HandDetector.is_pinch(r_lms))
    pinch_pos = HandDetector.pinch_point(r_lms, fw, fh) if pinching else None

    # if pinching we're dragging a slider, so don't count fingers (they're closed anyway)
    fingers = 0 if pinching else (HandDetector.full_finger_count(r_lms) if r_lms else 0)

    # finger count needs to be held steady for LOCK_HOLD seconds before it commits
    # otherwise quick passes through 1->2->3 fingers would flip the toggle constantly
    if fingers in (1, 2):
        if fingers != state.cand_count:
            state.cand_count = fingers
            state.cand_start = time.time()
        else:
            hold_prog = min((time.time() - state.cand_start) / LOCK_HOLD, 1.0)
            if hold_prog >= 1.0:
                state.reverb_on = (fingers == 2)
    else:
        state.cand_count = 0
        state.cand_start = None

    if pinching and pinch_pos:
        px, py = pinch_pos
        if not state.was_fist:
            # only latch a slider once at the moment the pinch starts
            state.grab = -1
            for idx, cx in enumerate(cxs):
                if abs(px - cx) < 90:
                    state.grab = idx
                    break
        if state.grab >= 0:
            val = max(0.0, min(1.0, (bot - py) / max(bot - top, 1))) 
            data[_SLIDER_KEYS[state.grab]] = val
    else:
        state.grab = -1

    state.was_fist = pinching
    return hold_prog


def draw(frame, fw: int, fh: int, data: dict, state: State, hold_prog: float = 0.0) -> None:
    top, bot, cxs = _layout(fw, fh)

    # ON/OFF pill toggle at the top
    pill_x1 = TAB_W + 20
    pill_x2 = fw - 20
    n_pills  = len(PILL_LABELS)
    seg_w    = (pill_x2 - pill_x1) // n_pills
    pill_h   = 28
    pill_y   = TITLE_H + 16

    active_pill = 0 if not state.reverb_on else 1
    if state.cand_count == 1:
        cand_pill = 0
    elif state.cand_count == 2:
        cand_pill = 1
    else:
        cand_pill = -1

    for i, name in enumerate(PILL_LABELS):
        px      = pill_x1 + i * seg_w
        pw      = seg_w - 6
        is_act  = (i == active_pill)
        is_cand = (i == cand_pill)
        is_off  = (i == 0)

        fill_col = C_ROSE if (is_off and is_act) else (C_PLUM if is_act else C_VOID)

        ov = frame.copy()
        cv2.rectangle(ov, (px, pill_y), (px + pw, pill_y + pill_h), fill_col, -1)
        cv2.addWeighted(frame, 0.65, ov, 0.35, 0, frame)

        border = C_AMBER if is_act else (C_ROSE if is_cand else C_PLUM)
        cv2.rectangle(frame, (px, pill_y), (px + pw, pill_y + pill_h),
                      border, 2 if is_act else 1, cv2.LINE_AA)

        short = f"{i + 1}: {name}"
        (lw, lh), _ = cv2.getTextSize(short, FONT, 0.38, 1)
        text_col = C_WHITE if is_act else (C_ROSE if is_off else C_PLUM)
        cv2.putText(frame, short,
                    (px + (pw - lw) // 2, pill_y + (pill_h + lh) // 2 - 2),
                    FONT, 0.38, text_col, 1, cv2.LINE_AA)

        if is_cand and hold_prog > 0 and not is_act:
            arc_cx = px + pw - 10
            arc_cy = pill_y + pill_h // 2
            cv2.ellipse(frame, (arc_cx, arc_cy), (8, 8),
                        -90, 0, int(360 * hold_prog), C_AMBER, 2, cv2.LINE_AA)

    size_val   = data["room_size"]
    decay_val  = data["decay"]
    size_pct   = f"{int(size_val  * 100)} %"
    decay_pct  = f"{int(decay_val * 100)} %"

    grabbed = (state.grab == 0)
    _v_slider(frame, cxs[0], top, bot, size_val,
              "Size", "0 %", "100 %", grabbed)
    cv2.putText(frame, size_pct,
                (cxs[0] + 28, int(bot - size_val * (bot - top)) + 5),
                FONT, 0.48, C_AMBER if grabbed else C_WHITE, 1, cv2.LINE_AA)

    grabbed = (state.grab == 1)
    _v_slider(frame, cxs[1], top, bot, decay_val,
              "Decay", "0 %", "100 %", grabbed)
    cv2.putText(frame, decay_pct,
                (cxs[1] + 28, int(bot - decay_val * (bot - top)) + 5),
                FONT, 0.48, C_AMBER if grabbed else C_WHITE, 1, cv2.LINE_AA)

    content_cx = TAB_W + (fw - TAB_W) // 2
    if state.reverb_on:
        readout = f"Size: {size_pct}     Decay: {decay_pct}"
        col     = C_WHITE
    else:
        readout = "REVERB OFF  —  hold 2 fingers to enable"
        col     = C_PLUM
    (rw, _), _ = cv2.getTextSize(readout, FONT, 0.50, 1)
    cv2.putText(frame, readout, (content_cx - rw // 2, bot + 38),
                FONT, 0.50, col, 1, cv2.LINE_AA)

    # cursor is drawn by the caller using r_palm
    if state.grab >= 0 and state.was_fist:
        pass
