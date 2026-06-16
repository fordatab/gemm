"""
Generate the 86x86 Fashion-MNIST TRAINING files that this assignment expects:
    data/train-86-images-idx3-ubyte
    data/train-86-labels-idx1-ubyte

The stock Fashion-MNIST train set is 28x28; the assignment uses an 86x86 resized
variant (the test set data/t10k-86-* is already 86x86). We download the official
Zalando train set, resize images 28->86, and write IDX files whose headers match
the test set's format: LITTLE-ENDIAN, magic number 0 (the reader's ReverseInt is
commented out, so it reads the header in native byte order).
"""
import gzip
import io
import os
import struct
import sys
import urllib.request

import numpy as np
from PIL import Image

OUT_DIR = "data"
SIZE = 86
MIRRORS = [
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]
# split -> (source images gz, source labels gz, output base names)
SPLITS = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "train-86-images-idx3-ubyte", "train-86-labels-idx1-ubyte"),
    "test":  ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
              "t10k-86-images-idx3-ubyte", "t10k-86-labels-idx1-ubyte"),
}


def download(name):
    last = None
    for base in MIRRORS:
        url = base + name
        try:
            print(f"Downloading {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa
            print(f"  failed: {e}")
            last = e
    raise RuntimeError(f"All mirrors failed for {name}: {last}")


def read_idx_images_28(raw_gz):
    with gzip.open(io.BytesIO(raw_gz)) as f:
        magic, n, rows, cols = struct.unpack(">4i", f.read(16))
        assert magic == 2051, magic
        assert (rows, cols) == (28, 28), (rows, cols)
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels(raw_gz):
    with gzip.open(io.BytesIO(raw_gz)) as f:
        magic, n = struct.unpack(">2i", f.read(8))
        assert magic == 2049, magic
        return np.frombuffer(f.read(n), dtype=np.uint8).copy()


def resize_to_86(imgs28):
    n = imgs28.shape[0]
    out = np.empty((n, SIZE, SIZE), dtype=np.uint8)
    for i in range(n):
        im = Image.fromarray(imgs28[i], mode="L").resize((SIZE, SIZE), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)
        if (i + 1) % 10000 == 0:
            print(f"  resized {i + 1}/{n}")
    return out


def write_images_le(path, imgs):
    n, rows, cols = imgs.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4i", 0, n, rows, cols))  # little-endian, magic 0
        f.write(imgs.tobytes())
    print(f"Wrote {path}  ({os.path.getsize(path)} bytes, {n} images {rows}x{cols})")


def write_labels_le(path, labels):
    n = labels.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<2i", 0, n))  # little-endian, magic 0
        f.write(labels.tobytes())
    print(f"Wrote {path}  ({os.path.getsize(path)} bytes, {n} labels)")


def gen_split(split):
    img_gz, lbl_gz, out_img, out_lbl = SPLITS[split]
    imgs28 = read_idx_images_28(download(img_gz))
    labels = read_idx_labels(download(lbl_gz))
    assert imgs28.shape[0] == labels.shape[0]
    print(f"[{split}] Loaded {imgs28.shape[0]} samples. Resizing 28->{SIZE} (bilinear) ...")
    imgs86 = resize_to_86(imgs28)
    write_images_le(os.path.join(OUT_DIR, out_img), imgs86)
    write_labels_le(os.path.join(OUT_DIR, out_lbl), labels)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Args: which split(s) to generate. Default: both.
    splits = [a for a in sys.argv[1:] if a in SPLITS] or ["train", "test"]
    for split in splits:
        gen_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
