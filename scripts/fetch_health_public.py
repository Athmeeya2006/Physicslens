"""Download the public machine-fault recordings used by the health_public case into var/health_public/ (gitignored).

  python scripts/fetch_health_public.py [bearings] [mimii] [spindown]     (default: all three)

Every file is checked against a sha256: the bearings against the hash Mendeley Data publishes for each file, the
spin-down record against the hash written below, and the MIMII clips against data/health_public/mimii_files.sha256
(written on the first fetch, checked on every later one). Nothing here is used at analysis time: the analysis reads
only the local copies.

Sizes: bearings 60 MATLAB files, about 310 MB. MIMII: 6 dB fan and pump, model IDs 00, 02, 04, 06; per ID 40
anomalous clips, 40 normal test clips and 60 normal reference clips (1,120 clips of 8-channel 16 kHz audio, about
2.9 GB), read straight out of the 10 GB Zenodo zips with HTTP range requests so the whole archive is never
downloaded. Spin-down: one 2 KB CSV.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var" / "health_public"
DATA = ROOT / "data" / "health_public"

# University of Ottawa UORED-VAFCLS, Mendeley Data, version 5 (CC BY 4.0).
# The unversioned endpoint lists every file of the latest version; the script stops if that is no longer version 5.
UORED_API = "https://data.mendeley.com/public-api/datasets/y2px5tg92h"
# MIMII, Zenodo record 3384388, "public 1.0" (CC BY-SA 4.0).
MIMII_URL = "https://zenodo.org/api/records/3384388/files/{name}/content"
MIMII_ZIPS = {"fan": "6_dB_fan.zip", "pump": "6_dB_pump.zip"}
MIMII_IDS = ("id_00", "id_02", "id_04", "id_06")
# Fixed before any score was seen; small enough to keep the download near 3 GB.
N_ANOM, N_NORM_TEST, N_NORM_REF = 40, 40, 60
MIMII_SEED = 20261102
MIMII_SHA = DATA / "mimii_files.sha256"
# Zenodo record 23231539 version 2 (data CC BY 4.0, scripts MIT).
SPIN_URL = "https://zenodo.org/api/records/23231539/files/bench_spindown_lifted.csv/content"
# Zenodo publishes md5 0fbefe8eca39d1049d8b7a52da456b63 for this file; the sha256 below matches that download.
SPINDOWN_SHA256 = "426c1aee42e64e9af3794629a504397a161840f968bc52b6f5577444851e209e"


def _get(url: str, headers: dict | None = None, tries: int = 12) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "physicslens-health-fetch", **(headers or {})})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == tries - 1:
                raise
            # Zenodo rate-limits per minute and says how long to wait; its gateways also drop out briefly.
            time.sleep(float(exc.headers.get("Retry-After") or 15 * (attempt + 1)) + 1)
        except (OSError, http.client.HTTPException) as exc:  # dropped connections and short reads on long runs
            if attempt == tries - 1:
                raise
            print(f"  retry after {exc}", file=sys.stderr)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError("unreachable")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_bearings() -> None:
    meta = json.loads(_get(UORED_API))
    if meta.get("version") != 5:
        raise SystemExit(f"Expected UORED-VAFCLS version 5, Mendeley returned version {meta.get('version')}.")
    dest = OUT / "uored"
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in meta["files"] if f["filename"].endswith(".mat")]
    for f in sorted(files, key=lambda f: f["filename"]):
        path = dest / f["filename"]
        want = f["content_details"]["sha256_hash"]
        if not (path.exists() and _sha256(path) == want):
            print("bearings:", f["filename"])
            path.write_bytes(_get(f["content_details"]["download_url"]))
        if _sha256(path) != want:
            raise SystemExit(f"sha256 mismatch for {path.name}")
    (dest / "mendeley_metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"bearings: {len(files)} files verified in {dest}")


# ---------- a minimal remote zip reader: only the central directory and the chosen members are fetched ----------

def _range(url: str, start: int, end: int) -> bytes:
    return _get(url, {"Range": f"bytes={start}-{end}"})


def _zip_directory(url: str, size: int) -> dict[str, tuple[int, int, int, int]]:
    """name -> (local header offset, compressed size, method, crc32). Handles ZIP64, which these >4 GB zips use."""
    tail = _range(url, size - 65536, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        loc = tail.rfind(b"PK\x06\x07")
        z64_off = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        z64 = _range(url, z64_off, z64_off + 55)
        cd_size, cd_off = struct.unpack("<QQ", z64[40:56])
    cd = _range(url, cd_off, cd_off + cd_size - 1)
    out, i = {}, 0
    while i < len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[i + 10:i + 12])
        crc, csize, usize = struct.unpack("<III", cd[i + 16:i + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[i + 28:i + 34])
        off, = struct.unpack("<I", cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nlen].decode()
        extra = cd[i + 46 + nlen:i + 46 + nlen + elen]
        # ZIP64 extra field: only the fields saturated in the fixed header are present, in this order.
        j = 0
        while j < len(extra):
            tag, ln = struct.unpack("<HH", extra[j:j + 4])
            if tag == 1:
                vals, k = extra[j + 4:j + 4 + ln], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", vals[k:k + 8])[0]; k += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", vals[k:k + 8])[0]
            j += 4 + ln
        out[name] = (off, csize, method, crc)
        i += 46 + nlen + elen + clen
    return out


def _zip_member(url: str, entry: tuple[int, int, int, int], name: str) -> bytes:
    """One range request per member: the local header is 30 bytes plus the name and a short extra field, so a
    generous margin covers it (Zenodo allows about two requests a second)."""
    off, csize, method, crc = entry
    blob = _range(url, off, off + 30 + len(name) + EXTRA_MARGIN + csize - 1)
    nlen, elen = struct.unpack("<HH", blob[26:30])
    if nlen + elen > len(name) + EXTRA_MARGIN:
        raise SystemExit(f"unexpected zip header for {name}")
    raw = blob[30 + nlen + elen:30 + nlen + elen + csize]
    data = raw if method == 0 else zlib.decompress(raw, -15)
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        raise SystemExit("CRC mismatch in a MIMII zip member")
    return data


EXTRA_MARGIN = 128


def mimii_selection(names: list[str], machine: str, mid: str) -> dict[str, list[str]]:
    """The fixed subset: a seeded shuffle of the sorted file names, so anyone can re-derive exactly the same clips."""
    rng = random.Random(f"{MIMII_SEED}-{machine}-{mid}")
    anom = sorted(n for n in names if f"/{mid}/abnormal/" in n and n.endswith(".wav"))
    norm = sorted(n for n in names if f"/{mid}/normal/" in n and n.endswith(".wav"))
    rng.shuffle(anom)
    rng.shuffle(norm)
    return {"anomalous": sorted(anom[:N_ANOM]), "normal_test": sorted(norm[:N_NORM_TEST]),
            "normal_ref": sorted(norm[N_NORM_TEST:N_NORM_TEST + N_NORM_REF])}


def fetch_mimii() -> None:
    known = {}
    if MIMII_SHA.exists():
        for line in MIMII_SHA.read_text().splitlines():
            h, name = line.split("  ", 1)
            known[name] = h
    lines, selection = [], {}
    for machine, zname in MIMII_ZIPS.items():
        url = MIMII_URL.format(name=zname)
        size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
                   .headers["Content-Length"])
        directory = _zip_directory(url, size)
        for mid in MIMII_IDS:
            sel = mimii_selection(list(directory), machine, mid)
            selection[f"{machine}/{mid}"] = sel
            names = [name for group in sel.values() for name in group]

            def one(name: str, url: str = url, directory: dict = directory) -> str:
                path = OUT / "mimii" / name
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(_zip_member(url, directory[name], name))
                    tmp.rename(path)  # a clip is either complete or absent, even if a run is interrupted
                h = _sha256(path)
                if name in known and known[name] != h:
                    raise SystemExit(f"sha256 mismatch for {name}")
                return f"{h}  {name}"

            # Each clip is three small range requests; a few in parallel keeps the run to minutes, not hours.
            with ThreadPoolExecutor(max_workers=2) as pool:
                lines.extend(pool.map(one, names))
            print(f"mimii: {machine} {mid} done")
    if not MIMII_SHA.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        MIMII_SHA.write_text("\n".join(lines) + "\n")
    (OUT / "mimii" / "selection.json").write_text(json.dumps(selection, indent=1))
    print(f"mimii: {len(lines)} clips verified")


def fetch_spindown() -> None:
    dest = OUT / "spindown" / "bench_spindown_lifted.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(SPIN_URL))
    h = _sha256(dest)
    if h != SPINDOWN_SHA256:
        raise SystemExit(f"sha256 mismatch for the spin-down record: {h}")
    print("spindown: verified")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["bearings", "mimii", "spindown"]
    for part in parts:
        {"bearings": fetch_bearings, "mimii": fetch_mimii, "spindown": fetch_spindown}[part]()
