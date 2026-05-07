import math
import socket
import struct

# everything gets squashed into 0-100 ints before sending to Max
_FREQ_MIN = 20.0
_FREQ_MAX = 22050.0
_GAIN_MIN = -20.0
_GAIN_MAX = 20.0
_Q_MIN = 0.1
_Q_MAX = 10.0


def _lin(v: float, lo: float, hi: float) -> int:
    return round(max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100)


def _log(v: float, lo: float, hi: float) -> int:
    # log scale for stuff like frequency where humans hear logarithmically
    if v <= 0.0:
        return 0
    return round(max(0.0, min(1.0, math.log10(v / lo) / math.log10(hi / lo))) * 100)


def _osc_str(s: str) -> bytes:
    # OSC strings: null-terminated, padded to 4 bytes. don't ask me why MAX requires 4
    b = s.encode("ascii") + b"\x00"
    pad = (4 - len(b) % 4) % 4
    return b + b"\x00" * pad


def _osc_int(address: str, value: int) -> bytes:
    # bare-minimum OSC packet: address + ",i" tag + int. that's all Max needs
    return _osc_str(address) + _osc_str(",i") + struct.pack(">i", value)


class UDPSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last: dict[str, int] = {}

    def _emit(self, key: str, val: int) -> None:
        # only send if the value actually changed, otherwise we spam Max for no reason
        if self._last.get(key) != val:
            self._last[key] = val
            address = f"/{key}"
            pkt = _osc_int(address, val)
            self._sock.sendto(pkt, self._addr)

    def send_channel(self, ch: int, volume: float, pan: float) -> None:
        self._emit(f"{ch}_vol", _lin(volume, 0.0, 1.0))
        self._emit(f"{ch}_pan", _lin(pan, 0.0, 1.0))

    def send_reverb(self, ch: int, reverb: dict, reverb_on: bool = False) -> None:
        # always send size + decay even when off, so Max has the latest values when re-enabled
        self._emit(f"{ch}_reverb_on", 1 if reverb_on else 0)
        self._emit(f"{ch}_reverb_size", _lin(reverb["room_size"], 0.0, 1.0))
        self._emit(f"{ch}_reverb_decay", _lin(reverb["decay"], 0.0, 1.0))

    def send_eq(self, ch: int, eq: dict, eq_on: bool, locked_type: int) -> None:
        # eq_mode: 0=off, 1=lowpass, 2=highpass, 3=bandpass, 4=bandstop
        mode = 0 if not eq_on else locked_type + 1
        self._emit(f"{ch}_eq_mode", mode)
        self._emit(f"{ch}_eq_freq", _log(eq["freq"], _FREQ_MIN, _FREQ_MAX))
        self._emit(f"{ch}_eq_gain", _lin(eq["gain"], _GAIN_MIN, _GAIN_MAX))
        self._emit(f"{ch}_eq_q", _log(eq["q"], _Q_MIN, _Q_MAX))

    def close(self) -> None:
        self._sock.close()
