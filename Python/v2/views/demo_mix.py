import os
import math
import time

os.environ["GLOG_minloglevel"] = os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2

from core.hand_detection import HandDetector
from core.udp import UDPSender
from core.gui import (
    dark_overlay,
    draw_title,
    C_VOID,
    C_PLUM,
    C_ROSE,
    C_AMBER,
    C_MAGENTA,
    C_WHITE,
    FONT,
    WIN_W,
    WIN_H,
)
from views.channel_mods import run_channel_mods, default_fx


MARGIN = 120
TITLE_H = 90
NAME_H = 40
KNOB_R = 30
KNOB_ZONE = KNOB_R * 2 + 36


def _draw_fader(
    frame, cx: int, top: int, bottom: int, volume: float, grabbed: bool
) -> None:
    track_w = 4
    cv2.line(frame, (cx, top), (cx, bottom), C_PLUM, track_w, cv2.LINE_AA)

    thumb_y = int(bottom - volume * (bottom - top))
    cv2.line(
        frame,
        (cx, thumb_y),
        (cx, bottom),
        C_AMBER if grabbed else C_ROSE,
        track_w,
        cv2.LINE_AA,
    )

    fill = C_MAGENTA if grabbed else C_PLUM
    border = C_AMBER if grabbed else C_WHITE
    cv2.rectangle(frame, (cx - 18, thumb_y - 8), (cx + 18, thumb_y + 8), fill, -1)
    cv2.rectangle(
        frame,
        (cx - 18, thumb_y - 8),
        (cx + 18, thumb_y + 8),
        border,
        1 + grabbed,
        cv2.LINE_AA,
    )

    # converting linear 0-1 volume into dB for the label, "audio engineer brain"
    db = 20 * math.log10(max(volume, 1e-4))
    lbl = "-inf" if volume < 0.01 else f"{db:+.0f}dB"
    cv2.putText(
        frame,
        lbl,
        (cx + 24, thumb_y + 5),
        FONT,
        0.38,
        C_WHITE if grabbed else C_PLUM,
        1,
        cv2.LINE_AA,
    )


def _draw_knob(frame, cx: int, cy: int, pan: float, active: bool) -> None:
    # arc goes from 120deg to 420deg (300deg sweep), looks like a real knob
    start_a = 120
    end_a = 120 + pan * 300

    cv2.ellipse(
        frame,
        (cx, cy),
        (KNOB_R, KNOB_R),
        0,
        start_a,
        start_a + 300,
        C_PLUM,
        2,
        cv2.LINE_AA,
    )
    if pan > 0.005:
        cv2.ellipse(
            frame,
            (cx, cy),
            (KNOB_R, KNOB_R),
            0,
            start_a,
            int(end_a),
            C_AMBER if active else C_ROSE,
            2,
            cv2.LINE_AA,
        )

    a_rad = math.radians(end_a)
    dx = int((KNOB_R - 6) * math.cos(a_rad))
    dy = int((KNOB_R - 6) * math.sin(a_rad))
    cv2.circle(
        frame, (cx + dx, cy + dy), 4, C_WHITE if active else C_ROSE, -1, cv2.LINE_AA
    )

    pan_v = pan * 2 - 1
    lbl = (
        "C"
        if abs(pan_v) < 0.05
        else (f"L{int(-pan_v * 100)}" if pan_v < 0 else f"R{int(pan_v * 100)}")
    )
    (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.38, 1)
    cv2.putText(
        frame,
        lbl,
        (cx - tw // 2, cy + KNOB_R + 18),
        FONT,
        0.38,
        C_WHITE if active else C_PLUM,
        1,
        cv2.LINE_AA,
    )


def _draw_channel(
    frame,
    ch_left: int,
    ch_w: int,
    fh: int,
    name: str,
    volume: float,
    pan: float,
    in_col: bool,
    fader_grabbed: bool,
    knob_grabbed: bool,
) -> None:
    cx = ch_left + ch_w // 2
    slider_top = TITLE_H + NAME_H + 15
    slider_bot = fh - KNOB_ZONE - 25
    knob_cy = fh - KNOB_ZONE // 2 - 10

    if in_col:
        ov = frame.copy()
        cv2.rectangle(
            ov, (ch_left + 2, TITLE_H - 5), (ch_left + ch_w - 2, fh - 10), C_PLUM, -1
        )
        cv2.addWeighted(frame, 0.88, ov, 0.12, 0, frame)

    (tw, _), _ = cv2.getTextSize(name, FONT, 0.55, 1)
    cv2.putText(
        frame,
        name,
        (cx - tw // 2, TITLE_H + NAME_H),
        FONT,
        0.55,
        C_WHITE if in_col else C_ROSE,
        1,
        cv2.LINE_AA,
    )

    _draw_fader(frame, cx, slider_top, slider_bot, volume, fader_grabbed)
    _draw_knob(frame, cx, knob_cy, pan, active=knob_grabbed)


def run_mix(
    num_channels: int = 4,
    names: list | None = None,
    udp_host: str = "127.0.0.1",
    udp_port: int = 9000,
) -> None:
    n = max(1, min(4, num_channels))
    labels = (list(names) if names else [f"Track {i + 1}" for i in range(n)])[:n]
    volumes = [0.30] * n
    pans = [0.50] * n
    fx_data = [default_fx() for _ in range(n)]

    udp = UDPSender(udp_host, udp_port)

    # blast initial state to Max so it knows the defaults
    for i in range(n):
        udp.send_channel(i + 1, volumes[i], pans[i])
        udp.send_reverb(i + 1, fx_data[i]["reverb"], reverb_on=False)
        udp.send_eq(i + 1, fx_data[i]["eq"], eq_on=False, locked_type=0)

    detector = HandDetector()
    shaka_start = None
    shaka_armed = False

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
    cv2.namedWindow("MOTION MIX", cv2.WINDOW_AUTOSIZE)
    t0 = time.time()

    grabbed_ch = -1
    was_pinching = False
    pinch_release_t = 0.0
    PINCH_RELEASE_GAP = 0.15

    # hysteresis: separate engage and release thresholds so the gesture doesn't flicker
    PINCH_ENGAGE  = 0.055
    PINCH_RELEASE = 0.080
    TWIST_ENGAGE  = 0.070
    TWIST_RELEASE = 0.100

    # exponential moving average smoothing. lower alpha = smoother but more lag
    VOL_ALPHA   = 0.30
    PAN_ALPHA   = 0.25
    ANGLE_ALPHA = 0.35

    l_hold_fingers = 0
    l_hold_start = None
    L_OPEN_HOLD = 1.5

    twist_ch = -1
    was_twisting = False
    twist_angle_ref = 0.0
    twist_pan_ref = 0.50
    twist_release_t = 0.0
    smooth_angle = 0.0
    TWIST_RELEASE_GAP = 0.20
    # 45deg of wrist rotation = full pan sweep. felt the most natural after lots of testing
    TWIST_SCALE = 4 / math.pi

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        dark_overlay(frame, strength=0.60)

        (palm_pos, fist, r_lms), (l_palm, l_fingers, l_lms) = detector.detect(
            frame, int(time.time() * 1000)
        )

        if r_lms:
            detector.draw_skeleton(frame, r_lms, fw, fh, C_PLUM, C_MAGENTA)
        if l_lms:
            detector.draw_skeleton(frame, l_lms, fw, fh, C_VOID, C_ROSE)

        right = r_lms is not None
        lms = r_lms

        # pinch (vol) and twist (pan) can both register at once because the geometry overlaps
        # so: whichever was already active wins; otherwise pick the tighter pinch
        if right:
            pd = math.sqrt((lms[4].x - lms[8].x) ** 2 + (lms[4].y - lms[8].y) ** 2)
            td = math.sqrt((lms[4].x - lms[12].x) ** 2 + (lms[4].y - lms[12].y) ** 2)
            pinching = pd < (PINCH_RELEASE if was_pinching else PINCH_ENGAGE)
            twisting = td < (TWIST_RELEASE if was_twisting else TWIST_ENGAGE)
            if pinching and twisting:
                if was_pinching and not was_twisting:
                    twisting = False
                elif was_twisting and not was_pinching:
                    pinching = False
                else:
                    if pd <= td:
                        twisting = False
                    else:
                        pinching = False
        else:
            pinching = twisting = False

        if right:
            if pinching:
                px, py = HandDetector.pinch_point(lms, fw, fh)
            else:
                px = int(lms[8].x * fw)
                py = int(lms[8].y * fh)
        else:
            px, py = 0, 0

        ch_w = (fw - 2 * MARGIN) // n
        slider_top = TITLE_H + NAME_H + 15
        slider_bot = fh - KNOB_ZONE - 25
        knob_cy = fh - KNOB_ZONE // 2 - 10

        now = time.time()

        # latch which channel got grabbed at the moment the pinch started
        if pinching and not was_pinching:
            grabbed_ch = -1
            pinch_release_t = 0.0
            for i in range(n):
                ch_left = MARGIN + i * ch_w
                if ch_left <= px < ch_left + ch_w and py < knob_cy - KNOB_R - 5:
                    grabbed_ch = i
                    break
        if not pinching:
            # don't release immediately, give a tiny grace period in case mediapipe blinked
            if grabbed_ch >= 0:
                if pinch_release_t == 0.0:
                    pinch_release_t = now
                elif now - pinch_release_t >= PINCH_RELEASE_GAP:
                    grabbed_ch = -1
                    pinch_release_t = 0.0
            else:
                pinch_release_t = 0.0
        else:
            pinch_release_t = 0.0
        was_pinching = pinching

        if twisting:
            if not was_twisting and twist_ch == -1:
                for i in range(n):
                    ch_left = MARGIN + i * ch_w
                    if ch_left <= px < ch_left + ch_w:
                        twist_ch = i
                        twist_angle_ref = HandDetector.twist_angle(lms)
                        smooth_angle = twist_angle_ref
                        twist_pan_ref = pans[i]
                        break
        else:
            if twist_ch >= 0:
                if twist_release_t == 0.0:
                    twist_release_t = now
                elif now - twist_release_t >= TWIST_RELEASE_GAP:
                    twist_ch = -1
                    twist_release_t = 0.0
            else:
                twist_release_t = 0.0
        was_twisting = twisting

        for i in range(n):
            ch_left = MARGIN + i * ch_w
            in_col = bool(right) and (ch_left <= px < ch_left + ch_w)

            if grabbed_ch == i and pinching:
                raw_vol = max(
                    0.0, min(1.0, (slider_bot - py) / max(slider_bot - slider_top, 1))
                )
                volumes[i] = VOL_ALPHA * raw_vol + (1.0 - VOL_ALPHA) * volumes[i]

            if twist_ch == i and twisting:
                raw_angle = HandDetector.twist_angle(lms)
                smooth_angle = ANGLE_ALPHA * raw_angle + (1.0 - ANGLE_ALPHA) * smooth_angle
                delta = smooth_angle - twist_angle_ref
                # wrap delta into [-pi, pi] so going past 180deg doesn't snap
                delta = (delta + math.pi) % (2 * math.pi) - math.pi
                raw_pan = max(0.0, min(1.0, twist_pan_ref + delta * TWIST_SCALE))
                pans[i] = PAN_ALPHA * raw_pan + (1.0 - PAN_ALPHA) * pans[i]

            udp.send_channel(i + 1, volumes[i], pans[i])

            _draw_channel(
                frame,
                ch_left,
                ch_w,
                fh,
                labels[i],
                volumes[i],
                pans[i],
                in_col,
                fader_grabbed=(grabbed_ch == i),
                knob_grabbed=(twist_ch == i),
            )

            if i < n - 1:
                sx = ch_left + ch_w
                cv2.line(
                    frame, (sx, TITLE_H - 5), (sx, fh - 10), C_PLUM, 1, cv2.LINE_AA
                )

        # left hand: hold N fingers up to open channel N's effects screen
        target_ch = l_fingers - 1
        l_prog = 0.0

        if 0 <= target_ch < n:
            if l_fingers == l_hold_fingers and l_hold_start is not None:
                l_prog = min((time.time() - l_hold_start) / L_OPEN_HOLD, 1.0)
                if l_prog >= 1.0:
                    chan_state = {"volume": volumes[target_ch], "pan": pans[target_ch]}
                    run_channel_mods(
                        cap,
                        detector,
                        labels[target_ch],
                        fx_data[target_ch],
                        chan_state,
                        initial_tab=0,
                        channel_idx=target_ch + 1,
                        udp=udp,
                    )
                    volumes[target_ch] = chan_state["volume"]
                    pans[target_ch] = chan_state["pan"]

                    # coming back from channel mods, reset everything so we don't
                    # accidentally have a stale grab
                    grabbed_ch = -1
                    was_pinching = False
                    pinch_release_t = 0.0
                    twist_ch = -1
                    was_twisting = False
                    twist_release_t = 0.0
                    smooth_angle = 0.0
                    shaka_start = None
                    l_hold_start = None
                    shaka_armed = False
                    l_hold_fingers = 0
                    l_prog = 0.0
            else:
                l_hold_fingers = l_fingers
                l_hold_start = time.time()
        else:
            l_hold_fingers = 0
            l_hold_start = None

        if l_prog > 0:
            ch_left = MARGIN + target_ch * ch_w
            cx_ch = ch_left + ch_w // 2
            cv2.ellipse(
                frame,
                (cx_ch, TITLE_H + NAME_H + 30),
                (16, 16),
                -90,
                0,
                int(360 * l_prog),
                C_AMBER,
                3,
                cv2.LINE_AA,
            )
            (tw, _), _ = cv2.getTextSize(labels[target_ch], FONT, 0.55, 1)
            bx = ch_left + (ch_w - tw) // 2
            cv2.putText(
                frame,
                labels[target_ch],
                (bx, TITLE_H + NAME_H),
                FONT,
                0.55,
                C_AMBER,
                1,
                cv2.LINE_AA,
            )

        if right:
            if twisting:
                detector.draw_twist(frame, lms, fw, fh, grabbed=(twist_ch >= 0))
            else:
                detector.draw_pinch(frame, lms, fw, fh, pinching)

        draw_title(frame, "MIX & MASTER", fw // 2, 52)

        # shaka_armed: don't trigger if a shaka was already up when we entered the screen
        shaka = bool(
            lms and not pinching and not twisting and HandDetector.is_shaka(lms)
        )
        if lms and not shaka:
            shaka_armed = True
        if shaka and shaka_armed:
            if shaka_start is None:
                shaka_start = time.time()
            shaka_prog = min((time.time() - shaka_start) / 0.8, 1.0)
            if shaka_prog >= 1.0:
                break
        else:
            shaka_start = None
            shaka_prog = 0.0

        col = C_AMBER if shaka_prog > 0 else C_PLUM
        hint = "R hand: PINCH = volume  |  TWIST (thumb+middle) = pan  |  hold shaka to go back"
        (hw, th), _ = cv2.getTextSize(hint, FONT, 0.38, 1)
        hx = (fw - hw) // 2
        hy = fh - 8
        cv2.putText(frame, hint, (hx, hy), FONT, 0.38, col, 1, cv2.LINE_AA)
        if shaka_prog > 0:
            cv2.ellipse(
                frame,
                (hx + hw + 16, hy - th // 2),
                (8, 8),
                -90,
                0,
                int(360 * shaka_prog),
                C_AMBER,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("MOTION MIX", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    udp.close()
