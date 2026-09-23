"""videos.py — tra cuu video theo nhan thay vi theo UUID."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "hyper-kvasir-videos" / "videos"
ANNOT = ROOT / "hyper-kvasir-videos" / "video-annotations.csv"

_cache = None


def load_annotations():
    global _cache
    if _cache is None:
        _cache = {}
        if ANNOT.exists():
            for line in ANNOT.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
                if ";" in line:
                    vid, finding = line.split(";", 1)
                    _cache[vid.strip()] = finding.strip()
    return _cache


def label_for(path):
    return load_annotations().get(Path(path).stem, "?")


def find(keyword=""):
    kw = keyword.lower()
    out = []
    for vid, finding in load_annotations().items():
        if kw and kw not in finding.lower() and kw not in vid.lower():
            continue
        p = VIDEO_DIR / f"{vid}.avi"
        if p.exists():
            out.append((vid, finding, p, p.stat().st_size / 1e6))
    return sorted(out, key=lambda t: t[1].lower())


def resolve(spec):
    if spec is None:
        return sorted(VIDEO_DIR.glob("*.avi"))[0]

    p = Path(spec)
    if p.exists() and p.is_file():
        return p

    hits = sorted(VIDEO_DIR.glob(f"{spec}*.avi"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SystemExit(f"'{spec}' khop {len(hits)} file, go them ky tu:\n  "
                         + "\n  ".join(h.name for h in hits[:10]))

    matches = find(spec)
    if len(matches) == 1:
        return matches[0][2]
    if len(matches) > 1:
        lines = "\n".join(f"  {v[:8]}  {f}" for v, f, _, _ in matches[:15])
        raise SystemExit(f"'{spec}' khop {len(matches)} video:\n{lines}\n"
                         f"-> go ro hon, hoac dung 8 ky tu dau cua id")
    raise SystemExit(f"Khong tim thay video nao khop '{spec}'.\n"
                     f"Thu:  python src/videos.py <tu khoa>")


def _print_table(rows, title):
    print(f"\n{title}  ({len(rows)} video)")
    print(f"{'id (8 ky tu)':14} {'kich thuoc':>11}  nhan")
    print("-" * 78)
    for vid, finding, _, mb in rows:
        print(f"{vid[:8]:14} {mb:8.1f} MB  {finding}")


if __name__ == "__main__":
    import sys
    kw = " ".join(sys.argv[1:])
    rows = find(kw)
    if not rows:
        print(f"Khong co video nao khop '{kw}'.")
    else:
        _print_table(rows, f"Tim '{kw}'" if kw else "Toan bo video co tren dia")
        print(f"\nChay:  python src/video_infer.py --video {rows[0][0][:8]} --n 400")
