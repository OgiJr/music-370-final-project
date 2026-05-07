import collections
import time

import cv2
import numpy as np

from core.hand_detection import HandDetector
from core.gui import (
    dark_overlay, draw_title,
    C_VOID, C_PLUM, C_ROSE, C_AMBER, C_MAGENTA, C_WHITE, FONT,
)
from views.channel_mods._layout import TITLE_H, HINT_H, TAB_W
from views.channel_mods import stereo_field, reverb as reverb_tab, eq as eq_tab
from core.udp import UDPSender

TABS       = ["Stereo Field", "Reverb", "EQ"]
SHAKA_HOLD = 0.8
TAB_HOLD   = 0.5


def default_fx() -> dict:
    return {
        "stereo": {"width": 0.60, "rotation": 0.50, "balance": 0.50},
        "reverb": {"room_size": 0.55, "decay": 0.45, "pre_delay": 0.20, "mix": 0.30},
        "eq":     {"type": 0, "freq": 1000.0, "gain": 0.0, "q": 1.0},
    }


def _draw_tab_col(
    frame,
    fh:        int,
    active:    int,
    target:    int,
    hold_prog: float,
) -> None:
    av_h  = fh - TITLE_H - HINT_H
    btn_h = av_h // len(TABS)

    for i, name in enumerate(TABS):
        by = TITLE_H + i * btn_h
        # last button takes up the leftover pixels so there's no gap
        bh = btn_h if i < len(TABS) - 1 else (fh - HINT_H - by)
        bh = max(bh, 1)

        is_act = (i == active)
        is_tgt = (i == target)

        roi  = frame[by:by + bh, 0:TAB_W]
        blr  = cv2.GaussianBlur(roi, (15, 15), 8)
        tint = np.full_like(blr, C_PLUM if is_act else C_VOID)
        frame[by:by + bh, 0:TAB_W] = cv2.addWeighted(blr, 0.55, tint, 0.45, 0)

        bc = C_AMBER if is_act else (C_ROSE if is_tgt else C_PLUM)
        cv2.rectangle(frame, (0, by), (TAB_W - 1, by + bh - 1),
                      bc, 2 if is_act else 1, cv2.LINE_AA)

        badge_cx, badge_cy = 36, by + bh // 2
        cv2.circle(frame, (badge_cx, badge_cy), 20,
                   C_AMBER if is_act else C_PLUM, -1, cv2.LINE_AA)
        fc = str(i + 1)
        (bw2, _), _ = cv2.getTextSize(fc, FONT, 0.75, 2)
        cv2.putText(frame, fc, (badge_cx - bw2 // 2, badge_cy + 9),
                    FONT, 0.75, C_WHITE, 2, cv2.LINE_AA)

        (tw2, _), _ = cv2.getTextSize(name, FONT, 0.62, 1)
        tx = 66 + (TAB_W - 76 - tw2) // 2
        cv2.putText(frame, name, (tx, by + bh // 2 + 8),
                    FONT, 0.62, C_WHITE if is_act else C_ROSE, 1, cv2.LINE_AA)

        if is_tgt and hold_prog > 0:
            cv2.ellipse(frame, (TAB_W - 22, by + bh // 2), (13, 13),
                        -90, 0, int(360 * hold_prog), C_AMBER, 2, cv2.LINE_AA)

    cv2.line(frame, (TAB_W, TITLE_H), (TAB_W, fh - HINT_H), C_PLUM, 1, cv2.LINE_AA)


def run_channel_mods(
    cap,
    detector:     HandDetector,
    channel_name: str,
    fx_data:      dict | None = None,
    chan_state:   dict | None = None,
    initial_tab:  int = 0,
    channel_idx:  int = 1,
    udp:          "UDPSender | None" = None,
) -> None:
    if fx_data is None:
        fx_data = default_fx()
    if chan_state is None:
        chan_state = {"volume": 0.80, "pan": 0.50}

    active_tab    = max(0, min(2, initial_tab))
    shaka_start   = None
    # ignore any shaka leftover from when the previous screen exited
    shaka_armed   = False
    tab_hold_fing = -1
    tab_hold_t    = None

    stereo_trail  = collections.deque(maxlen=stereo_field.TRAIL_LEN)
    rv_state      = reverb_tab.State()
    eq_state      = eq_tab.State(locked_type=fx_data["eq"]["type"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        dark_overlay(frame, strength=0.65)

        (r_palm, r_fist, r_lms), (l_palm, l_fingers, l_lms) = \
            detector.detect(frame, int(time.time() * 1000))

        if r_lms:
            detector.draw_skeleton(frame, r_lms, fw, fh, C_PLUM, C_MAGENTA)
        if l_lms:
            detector.draw_skeleton(frame, l_lms, fw, fh, C_VOID, C_ROSE)

        # left hand: 1/2/3 fingers held = switch to that tab
        target    = l_fingers - 1 if l_fingers in (1, 2, 3) else -1
        hold_prog = 0.0

        if target >= 0 and target != active_tab:
            if tab_hold_fing == target and tab_hold_t is not None:
                hold_prog = min((time.time() - tab_hold_t) / TAB_HOLD, 1.0)
                if hold_prog >= 1.0:
                    active_tab    = target
                    tab_hold_fing = -1
                    tab_hold_t    = None
                    hold_prog     = 0.0
                    stereo_trail.clear()
            else:
                tab_hold_fing = target
                tab_hold_t    = time.time()
        else:
            tab_hold_fing = -1
            tab_hold_t    = None

        shaka = bool(r_lms and HandDetector.is_shaka(r_lms))
        if r_lms and not shaka:
            shaka_armed = True
        if shaka and shaka_armed:
            shaka_start = shaka_start or time.time()
            shaka_prog  = min((time.time() - shaka_start) / SHAKA_HOLD, 1.0)
            if shaka_prog >= 1.0:
                break
        else:
            shaka_start = None
            shaka_prog  = 0.0

        eq_hold_prog = 0.0
        rv_hold_prog = 0.0
        l_fist       = (l_lms is not None and l_fingers == 0)

        # each tab has its own update() that mutates fx_data in place
        if active_tab == 0:
            pass
        elif active_tab == 1:
            rv_hold_prog = reverb_tab.update(
                fx_data["reverb"], r_fist, r_palm, r_lms, fw, fh, rv_state,
            )
            if udp:
                udp.send_reverb(channel_idx, fx_data["reverb"], rv_state.reverb_on)
        elif active_tab == 2:
            eq_hold_prog = eq_tab.update(
                fx_data["eq"], r_fist, r_palm, r_lms,
                l_fist, l_palm, fw, fh, eq_state,
            )
            if udp:
                udp.send_eq(channel_idx, fx_data["eq"], eq_state.eq_on, eq_state.locked_type)

        if active_tab != 1:
            # if you switched tabs mid-grab, drop the grab so it doesn't keep dragging
            rv_state.grab     = -1
            rv_state.was_fist = False

        _draw_tab_col(frame, fh, active_tab, tab_hold_fing, hold_prog)

        if active_tab == 0:
            stereo_field.draw(frame, fw, fh, chan_state, r_palm, stereo_trail)
            if udp:
                udp.send_channel(channel_idx, chan_state["volume"], chan_state["pan"])

        elif active_tab == 1:
            reverb_tab.draw(frame, fw, fh, fx_data["reverb"], rv_state, rv_hold_prog)
            if r_lms and HandDetector.is_pinch(r_lms):
                pinch_pt = HandDetector.pinch_point(r_lms, fw, fh)
                col = C_AMBER if rv_state.grab >= 0 else C_PLUM
                cv2.circle(frame, pinch_pt, 18, col, 3, cv2.LINE_AA)

        else:
            eq_tab.draw(
                frame, fw, fh, fx_data["eq"],
                r_palm, l_palm, r_fist, l_fist,
                eq_state, eq_hold_prog, r_lms,
            )

        draw_title(
            frame,
            f"{channel_name.upper()} : {TABS[active_tab].upper()}",
            TAB_W + (fw - TAB_W) // 2,
            58,
        )

        col  = C_AMBER if shaka_prog > 0 else C_PLUM
        if active_tab == 2:
            hint = "R fingers 1-5 = filter type   |   R pinch: X=cutoff Y=gain   |   L fist Y=Q   |   shaka = back"
        elif active_tab == 1:
            hint = "R hand: 1 finger = OFF   2+ fingers = ON   |   pinch = drag sliders   |   shaka = back"
        else:
            hint = "L hand: 1 = Stereo Field   2 = Reverb   3 = EQ   |   hold shaka (R) to go back"
        (hw, th), _ = cv2.getTextSize(hint, FONT, 0.40, 1)
        hx = (fw - hw) // 2
        hy = fh - 12
        cv2.putText(frame, hint, (hx, hy), FONT, 0.40, col, 1, cv2.LINE_AA)
        if shaka_prog > 0:
            cv2.ellipse(frame, (hx + hw + 16, hy - th // 2), (9, 9),
                        -90, 0, int(360 * shaka_prog), C_AMBER, 2, cv2.LINE_AA)

        cv2.imshow("MOTION MIX", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
