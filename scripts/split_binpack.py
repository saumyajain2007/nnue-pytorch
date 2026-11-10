"""Split a (possibly truncated) .binpack into train/val files at chunk boundaries.

Binpack layout: repeated [b"BINP"][u32 little-endian payload size][payload].
Chunks are self-contained, so any subset of whole chunks is a valid binpack.
A trailing partial chunk (e.g. from an HTTP Range download) is dropped.

The validation set is NUM_VAL consecutive chunks starting at chunk VAL_START
(default: the last NUM_VAL chunks); every other chunk goes to the train file.

Usage: split_binpack.py SRC TRAIN_OUT VAL_OUT --num-val N [--val-start S]
"""

import argparse
import struct


def scan_chunks(f, size):
    chunks = []  # (offset, total_len)
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
    return chunks, off


def copy_ranges(f, out_path, ranges):
    total = 0
    with open(out_path, "wb") as out:
        for start, length in ranges:
            f.seek(start)
            remaining = length
            while remaining:
                buf = f.read(min(remaining, 1 << 24))
                out.write(buf)
                remaining -= len(buf)
            total += length
    return total


def merge(chunks):
    """Merge adjacent chunks into contiguous byte ranges."""
    ranges = []
    for off, ln in chunks:
        if ranges and ranges[-1][0] + ranges[-1][1] == off:
            ranges[-1] = (ranges[-1][0], ranges[-1][1] + ln)
        else:
            ranges.append((off, ln))
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("train_out")
    ap.add_argument("val_out")
    ap.add_argument("--num-val", type=int, required=True)
    ap.add_argument("--val-start", type=int, default=None)
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunks, used = scan_chunks(f, size)
        print(f"{len(chunks)} complete chunks, {used} of {size} bytes used")
        start = len(chunks) - args.num_val if args.val_start is None else args.val_start
        end = start + args.num_val
        assert 0 <= start < end <= len(chunks), (start, end, len(chunks))
        val = chunks[start:end]
        train = chunks[:start] + chunks[end:]
        for path, sel in ((args.train_out, train), (args.val_out, val)):
            n = copy_ranges(f, path, merge(sel))
            print(f"{path}: {len(sel)} chunks, {n} bytes")


if __name__ == "__main__":
    main()
