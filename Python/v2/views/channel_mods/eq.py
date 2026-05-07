import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from core.gui import C_VOID, C_PLUM, C_ROSE, C_AMBER, C_MAGENTA, C_WHITE, FONT
from core.hand_detection import HandDetector
from views.channel_mods._layout import TITLE_H, HINT_H, TAB_W

LOCK_HOLD      = 0.80
# little cooldown after pinch release so finger-counting doesn't accidentally fire
# the moment you let go (your hand always passes through "5 fingers" as it opens)
PINCH_COOLDOWN = 1.20

FREQ_MIN  = 20.0
FREQ_MAX  = 22050.0
GAIN_MIN  = -20.0
GAIN_MAX  =  20.0
DB_MIN    = -30.0
DB_MAX    =  30.0
FS        = 44100
N_PTS     = 220

PILL_LABELS  = ["OFF", "Lowpass", "Highpass", "Bandpass", "Band Stop"]
FILTER_TYPES = PILL_LABELS[1:]


@dataclass
class State:
    eq_on:          bool          = False
    locked_type:    int           = 0
    cand_count:     int           = 0
    cand_start:     float | None  = None
    was_pinching:   bool          = False
    pinch_end_t:    float         = 0.0


def _biquad(type_idx: int, freq_hz: float, q: float):
    # standard biquad coefficients from the RBJ audio EQ cookbook
    # (look it up if you actually want to understand this, I sure didn't)
    freq_hz = max(FREQ_MIN, min(freq_hz, FS / 2 - 1))
    q       = max(0.05, q)
    w0      = 2 * math.pi * freq_hz / FS
    c, s    = math.cos(w0), math.sin(w0)
    alpha   = s / (2.0 * q)

    if type_idx == 0:
        b = np.array([(1 - c) / 2,  1 - c,       (1 - c) / 2])
        a = np.array([ 1 + alpha,   -2 * c,        1 - alpha  ])
    elif type_idx == 1:
        b = np.array([(1 + c) / 2, -(1 + c),     (1 + c) / 2])
        a = np.array([ 1 + alpha,   -2 * c,        1 - alpha  ])
    elif type_idx == 2:
        b = np.array([alpha,         0.0,          -alpha     ])
        a = np.array([1 + alpha,    -2 * c,         1 - alpha ])
    else:
        b = np.array([1.0,          -2 * c,         1.0       ])
        a = np.array([1 + alpha,    -2 * c,         1 - alpha ])

    a0 = a[0]
    return b / a0, np.array([1.0, a[1] / a0, a[2] / a0])


def _freq_response_db(b, a, freqs) -> np.ndarray:
    # plug each frequency into H(z) = b(z)/a(z), take magnitude, convert to dB
    w    = 2 * np.pi * np.asarray(freqs, dtype=np.float64) / FS
    ejw  = np.exp(-1j * w)
    ejw2 = ejw * ejw
    H    = (b[0] + b[1] * ejw + b[2] * ejw2) / (1.0 + a[1] * ejw + a[2] * ejw2)
    return 20.0 * np.log10(np.maximum(np.abs(H), 1e-10))


def _graph_bounds(fw: int, fh: int) -> tuple[int, int, int, int]:
    return TAB_W + 58, fw - 36, TITLE_H + 62, fh - HINT_H - 52


def _freq_to_x(freq: float, gx1: int, gx2: int) -> int:
    # log scale because human hearing is logarithmic, otherwise the bass takes 90% of the screen
    freq = max(FREQ_MIN, min(freq, FREQ_MAX))
    t    = math.log10(freq / FREQ_MIN) / math.log10(FREQ_MAX / FREQ_MIN)
    return int(gx1 + t * (gx2 - gx1))


def _db_to_y(db: float, gy1: int, gy2: int) -> int:
    db = max(DB_MIN, min(db, DB_MAX))
    t  = (DB_MAX - db) / (DB_MAX - DB_MIN)
    return int(max(gy1, min(gy2, gy1 + t * (gy2 - gy1))))


def update(
    data:   dict,
    r_fist: bool,
    r_palm,
    r_lms,
    l_fist: bool,
    l_palm,
    fw:     int,
    fh:     int,
    state:  State,
) -> float:
    gx1, gx2, gy1, gy2 = _graph_bounds(fw, fh)
    hold_prog = 0.0

    pinching  = bool(r_lms and HandDetector.is_pinch(r_lms))
    pinch_pos = HandDetector.pinch_point(r_lms, fw, fh) if pinching else None

    # remember when the pinch released so we can ignore finger counts for a bit
    if state.was_pinching and not pinching:
        state.pinch_end_t = time.time()
    state.was_pinching = pinching

    pinch_cooldown_active = (
        state.pinch_end_t > 0.0
        and (time.time() - state.pinch_end_t) < PINCH_COOLDOWN
    )

    # finger counting only valid if: hand visible, not pinching, not on cooldown,
    # not throwing a shaka, and all fingertips are actually inside the frame
    fingers_valid = (
        r_lms is not None
        and not pinching
        and not pinch_cooldown_active
        and not HandDetector.is_shaka(r_lms)
        and HandDetector.all_fingertips_visible(r_lms)
    )
    eq_r_fingers = HandDetector.full_finger_count(r_lms) if fingers_valid else 0

    if eq_r_fingers >= 1:
        if eq_r_fingers != state.cand_count:
            state.cand_count = eq_r_fingers
            state.cand_start = time.time()
        else:
            hold_prog = min((time.time() - state.cand_start) / LOCK_HOLD, 1.0)
            if hold_prog >= 1.0:
                if eq_r_fingers == 1:
                    state.eq_on = False
                else:
                    # 2 fingers = first filter type, 3 = second, etc
                    state.eq_on       = True
                    state.locked_type = min(eq_r_fingers - 2, len(FILTER_TYPES) - 1)
                    data["type"]      = state.locked_type
    else:
        state.cand_count = 0
        state.cand_start = None

    # right pinch position = freq (X log scale) + gain (Y linear)
    if pinching and pinch_pos:
        px_n         = max(0.0, min(1.0, (pinch_pos[0] - gx1) / max(gx2 - gx1, 1)))
        data["freq"] = FREQ_MIN * (FREQ_MAX / FREQ_MIN) ** px_n
        py_n         = max(0.0, min(1.0, (pinch_pos[1] - gy1) / max(gy2 - gy1, 1)))
        data["gain"] = GAIN_MAX - py_n * (GAIN_MAX - GAIN_MIN)

    # left fist Y = Q (resonance), log scale from 0.1 to 10
    if l_fist and l_palm:
        ly_n      = max(0.0, min(1.0, (l_palm[1] - gy1) / max(gy2 - gy1, 1)))
        data["q"] = 10.0 ** (1.0 - 2.0 * ly_n)

    return hold_prog


def draw(
    frame,
    fw:        int,
    fh:        int,
    data:      dict,
    r_palm,
    l_palm,
    r_fist:    bool,
    l_fist:    bool,
    state:     State,
    hold_prog: float,
    r_lms=None,
) -> None:
    gx1, gx2, gy1, gy2 = _graph_bounds(fw, fh)

    active_pill = 0 if not state.eq_on else state.locked_type + 1
    cand_pill   = state.cand_count - 1 if state.cand_count >= 1 else -1

    n_pills = len(PILL_LABELS)
    seg_w   = (gx2 - gx1) // n_pills
    pill_h  = 28
    pill_y  = TITLE_H + 16

    for i, name in enumerate(PILL_LABELS):
        px      = gx1 + i * seg_w
        pw      = seg_w - 6
        is_act  = (i == active_pill)
        is_cand = (i == cand_pill)
        is_off  = (i == 0)

        fill_col  = C_PLUM if is_act else C_VOID
        if is_off and is_act:
            fill_col = C_ROSE

        ov = frame.copy()
        cv2.rectangle(ov, (px, pill_y), (px + pw, pill_y + pill_h), fill_col, -1)
        cv2.addWeighted(frame, 0.65, ov, 0.35, 0, frame)

        border = C_AMBER if is_act else (C_ROSE if is_cand else C_PLUM)
        cv2.rectangle(frame, (px, pill_y), (px + pw, pill_y + pill_h),
                      border, 2 if is_act else 1, cv2.LINE_AA)

        short = f"{i + 1}: {'OFF' if is_off else name[:4]}"
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

    ov = frame.copy()
    cv2.rectangle(ov, (gx1, gy1), (gx2, gy2), (8, 4, 18), -1)
    cv2.addWeighted(frame, 0.60, ov, 0.40, 0, frame)

    for db in [-24, -12, 0, 12, 24]:
        y   = _db_to_y(db, gy1, gy2)
        col = C_ROSE if db == 0 else C_PLUM
        cv2.line(frame, (gx1, y), (gx2, y), col, 2 if db == 0 else 1, cv2.LINE_AA)
        lbl = f"{db:+d}"
        (lw, lh), _ = cv2.getTextSize(lbl, FONT, 0.34, 1)
        cv2.putText(frame, lbl, (gx1 - lw - 4, y + lh // 2),
                    FONT, 0.34, C_PLUM, 1, cv2.LINE_AA)

    for f_hz in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        x   = _freq_to_x(f_hz, gx1, gx2)
        cv2.line(frame, (x, gy1), (x, gy2), C_PLUM, 1, cv2.LINE_AA)
        lbl = f"{f_hz // 1000}k" if f_hz >= 1000 else str(f_hz)
        (lw, _), _ = cv2.getTextSize(lbl, FONT, 0.34, 1)
        cv2.putText(frame, lbl, (x - lw // 2, gy2 + 18),
                    FONT, 0.34, C_PLUM, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), C_PLUM, 1, cv2.LINE_AA)

    zero_y = _db_to_y(0.0, gy1, gy2)

    if not state.eq_on:
        # EQ off: just draw a flat line at 0dB
        cv2.line(frame, (gx1, zero_y), (gx2, zero_y), C_PLUM,    1, cv2.LINE_AA)
        cv2.line(frame, (gx1, zero_y), (gx2, zero_y), C_MAGENTA, 2, cv2.LINE_AA)
    else:
        # compute the actual filter curve from the biquad coefficients
        b, a    = _biquad(state.locked_type, data["freq"], data["q"])
        freqs   = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_PTS)
        mags_db = _freq_response_db(b, a, freqs) + data["gain"]

        pts = [(_freq_to_x(f, gx1, gx2), _db_to_y(db, gy1, gy2))
               for f, db in zip(freqs, mags_db)]

        # fill the area under the curve down to 0dB so it looks like a real EQ plot
        fill_pts = pts + [(pts[-1][0], zero_y), (pts[0][0], zero_y)]
        ov_fill  = frame.copy()
        cv2.fillPoly(ov_fill, [np.array(fill_pts, dtype=np.int32)], C_PLUM)
        cv2.addWeighted(frame, 0.74, ov_fill, 0.26, 0, frame)

        # draw the line twice (thick magenta then thinner amber) for a glow effect
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], C_MAGENTA, 4, cv2.LINE_AA)
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], C_AMBER,   2, cv2.LINE_AA)

        fx    = _freq_to_x(data["freq"], gx1, gx2)
        gy_pt = _db_to_y(data["gain"],  gy1, gy2)
        cv2.line(frame, (fx, gy1), (fx, gy2), C_AMBER, 1, cv2.LINE_AA)
        cv2.circle(frame, (fx, gy_pt), 8,  C_AMBER, -1, cv2.LINE_AA)
        cv2.circle(frame, (fx, gy_pt), 13, C_WHITE,  1, cv2.LINE_AA)

    if l_palm and l_fist:
        ly = max(gy1, min(gy2, l_palm[1]))
        cv2.line(frame,   (gx1, ly), (gx2, ly), C_ROSE, 1, cv2.LINE_AA)
        cv2.circle(frame, (gx1 - 9, ly), 6, C_ROSE, -1, cv2.LINE_AA)
        cv2.putText(frame, f"Q {data['q']:.2f}", (gx1 + 6, ly - 6),
                    FONT, 0.44, C_ROSE, 1, cv2.LINE_AA)

    if state.eq_on:
        fq       = data["freq"]
        freq_str = f"{fq:.0f} Hz" if fq < 1000 else f"{fq / 1000:.2f} kHz"
        readout  = f"Freq: {freq_str}     Gain: {data['gain']:+.1f} dB     Q: {data['q']:.2f}"
    else:
        readout  = "EQ OFF  —  hold 2–5 fingers to select filter"

    (rw, _), _ = cv2.getTextSize(readout, FONT, 0.50, 1)
    content_cx = TAB_W + (fw - TAB_W) // 2
    cv2.putText(frame, readout, (content_cx - rw // 2, gy2 + 38),
                FONT, 0.50, C_WHITE if state.eq_on else C_PLUM, 1, cv2.LINE_AA)

    if r_lms and HandDetector.is_pinch(r_lms):
        pinch_pt = HandDetector.pinch_point(r_lms, fw, fh)
        cv2.circle(frame, pinch_pt, 18, C_AMBER, 3, cv2.LINE_AA)
    elif r_palm:
        cv2.circle(frame, r_palm, 10, C_PLUM, 2, cv2.LINE_AA)

    if l_palm and l_fist:
        cv2.circle(frame, l_palm, 18, C_ROSE, 3, cv2.LINE_AA)
