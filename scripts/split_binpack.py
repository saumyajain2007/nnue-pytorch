"""Split a (possibly truncated) .binpack into train/val files at chunk boundaries.

Binpack layout: repeated [b"BINP"][u32 little-endian payload size][payload].
A trailing partial chunk (from a Range download) is dropped.
Usage: split_binpack.py SRC TRAIN_OUT VAL_OUT NUM_VAL_CHUNKS
"""

import struct
import sys

src, train_out, val_out, n_val = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

chunks = []  # (offset, total_len)
with open(src, "rb") as f:
    f.seek(0, 2)
    size = f.tell()
    off = 0
    while off + 8 <= size:
        f.seek(off)
        hdr = f.read(8)
        if hdr[:4] != b"BINP":
            raise SystemExit(f"bad chunk header at {off}: {hdr[:4]!r}")
        n = struct.unpack("<I", hdr[4:])[0]
        if off + 8 + n > size:
            break  # truncated final chunk
        chunks.append((off, 8 + n))
        off += 8 + n

    print(f"{len(chunks)} complete chunks, {off} of {size} bytes used")
    assert len(chunks) > n_val * 2
    split = len(chunks) - n_val
    for path, sel in ((train_out, chunks[:split]), (val_out, chunks[split:])):
        with open(path, "wb") as out:
            start = sel[0][0]
            end = sel[-1][0] + sel[-1][1]
            f.seek(start)
            remaining = end - start
            while remaining:
                buf = f.read(min(remaining, 1 << 24))
                out.write(buf)
                remaining -= len(buf)
        print(f"{path}: {len(sel)} chunks, {end - start} bytes")
