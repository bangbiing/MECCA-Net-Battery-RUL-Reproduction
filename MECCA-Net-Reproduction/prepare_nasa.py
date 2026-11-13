"""Download the four NASA PCoE cells used by MECCA-Net and build NASA.npy.

Output format expected by mecca_net_nasa_autoregressive.py:
    {"B0005": {"capacity": np.ndarray}, ...}

Source: https://github.com/changyeon99/Battery-Data-Set (mirror of the NASA
PCoE "Battery Data Set"), downloaded through the jsDelivr CDN because
raw.githubusercontent.com is not reachable from this network.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request

import numpy as np
import scipy.io as sio

BASE = "https://cdn.jsdelivr.net/gh/changyeon99/Battery-Data-Set@main/"
NAMES = ["B0005", "B0006", "B0007", "B0018"]
RAW_DIR = os.path.join("data", "NASA", "raw")
OUT = os.path.join("data", "NASA", "NASA.npy")


def download(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.mat")
    url = BASE + f"{name}.mat"
    tmp = path + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    for attempt in range(6):
        try:
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
            os.replace(tmp, path)
            print(f"[ok] {name}: {os.path.getsize(path)/1e6:.1f} MB in {time.time()-t0:.0f}s")
            return path
        except Exception as exc:  # noqa: BLE001
            print(f"[retry {attempt}] {name}: {exc}")
            time.sleep(3)
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    raise RuntimeError(f"failed to download {name}")


def discharge_capacities(path: str) -> np.ndarray:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    key = [k for k in mat if not k.startswith("__")][0]
    cycles = mat[key].cycle
    caps = []
    for cyc in np.atleast_1d(cycles):
        if str(getattr(cyc, "type", "")) != "discharge":
            continue
        try:
            cap = float(cyc.data.Capacity)
        except AttributeError:
            continue
        if np.isfinite(cap) and cap > 0:
            caps.append(cap)
    return np.asarray(caps, dtype=np.float32)


def main() -> None:
    data = {}
    for name in NAMES:
        path = download(name)
        caps = discharge_capacities(path)
        data[name] = {"capacity": caps}
        print(f"  {name}: {len(caps)} discharge cycles, {caps[0]:.3f} -> {caps[-1]:.3f} Ah")
    np.save(OUT, data, allow_pickle=True)
    print("saved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    sys.exit(main())
