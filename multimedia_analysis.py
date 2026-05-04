# -*- coding: utf-8 -*-
"""
=============================================================================
  Multimedia Project - Huffman Coding Compression & Classification Analysis
=============================================================================
  Author  : Mohamed Taha
  Date    : 2026
  Purpose : Classify skin-disease images via Hugging Face API, apply
            Huffman Coding (lossless) compression per colour channel,
            and evaluate quality metrics (PSNR, MSE, Compression Ratio,
            File Size).

  Huffman Coding Overview
  -----------------------
  1. Count the frequency of every pixel value (0-255) in a channel.
  2. Build a min-heap (priority queue) from the frequency table.
  3. Repeatedly merge the two lowest-frequency nodes until one root
     remains -> this is the Huffman tree.
  4. Traverse the tree to assign a unique binary code to each symbol.
  5. Encode all pixels using those codes -> produces a bit-string.
  6. Decode the bit-string by walking the tree -> original pixels.
  Because no information is discarded, the method is LOSSLESS:
      MSE  = 0.0   (perfect reconstruction)
      PSNR = inf   (no noise introduced)

  Compression Ratio = (original bits) / (Huffman-encoded bits)
  A value > 1 means the Huffman stream is smaller than raw pixels.

=============================================================================

  Dependencies (install once):
      pip install gradio_client opencv-python numpy

  Directory Layout Expected:
      Muiltmedia_project/
      +-- Measles/
      |   +-- measles10.png
      |   +-- measles11.png
      +-- Monkeypox/
      |   +-- monkeypox179.png
      |   +-- monkeypox241.png
      +-- multimedia_analysis.py   <- this script
=============================================================================
"""

# --- Standard & Third-party Imports -----------------------------------------
import os
import sys
import math
import json
import time
import heapq
import warnings
from collections import Counter

import cv2
import numpy as np

# Force UTF-8 output so characters print correctly on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Gradio client to reach the Hugging Face Space API
from gradio_client import Client, handle_file

warnings.filterwarnings("ignore")


# --- Configuration -----------------------------------------------------------
HF_API_URL = "https://m-taha6-monkeypox.hf.space/"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CLASSES    = ["Measles", "Monkeypox"]   # subfolder names == class labels


# =============================================================================
#  HUFFMAN CODING IMPLEMENTATION
# =============================================================================

class HuffmanNode:
    """
    A single node in the Huffman binary tree.

    Attributes
    ----------
    symbol : int or None
        Pixel intensity value (0-255) for leaf nodes; None for internal nodes.
    freq : int
        Cumulative frequency of all symbols under this subtree.
    left, right : HuffmanNode or None
        Child nodes; both None for leaf nodes.
    """

    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq   = freq
        self.left   = None
        self.right  = None

    # heapq compares nodes by (freq, ...) - define < to avoid errors on tie
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_table: dict) -> HuffmanNode:
    """
    Construct a Huffman tree from a symbol->frequency mapping.

    Algorithm
    ---------
    1. Create one leaf node per symbol and push all leaves onto a min-heap.
    2. While the heap has more than one node:
       a. Pop the two nodes with the lowest frequency (n1, n2).
       b. Create an internal node whose frequency = n1.freq + n2.freq,
          with n1 as the left child and n2 as the right child.
       c. Push the new internal node back onto the heap.
    3. The single remaining node is the tree root.

    Parameters
    ----------
    freq_table : dict {symbol: frequency}

    Returns
    -------
    HuffmanNode  - root of the complete Huffman tree
    """
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_table.items()]
    heapq.heapify(heap)

    # Edge case: only one unique symbol in the channel
    if len(heap) == 1:
        root        = HuffmanNode(None, heap[0].freq)
        root.left   = heapq.heappop(heap)
        return root

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)

        merged       = HuffmanNode(None, n1.freq + n2.freq)
        merged.left  = n1
        merged.right = n2

        heapq.heappush(heap, merged)

    return heap[0]   # root


def generate_codes(node: HuffmanNode,
                   prefix: str = "",
                   codes: dict = None) -> dict:
    """
    Recursively traverse the Huffman tree to build the code table.

    Left edges are labelled '0', right edges '1'.
    Leaf nodes receive the accumulated prefix as their codeword.

    Parameters
    ----------
    node   : HuffmanNode - current tree node
    prefix : str         - binary string accumulated so far
    codes  : dict        - output {symbol: binary_string}

    Returns
    -------
    dict {symbol: binary_string}
    """
    if codes is None:
        codes = {}

    if node is None:
        return codes

    # Leaf node - assign the prefix as this symbol's codeword
    if node.symbol is not None:
        codes[node.symbol] = prefix if prefix else "0"   # handle single-symbol tree
        return codes

    generate_codes(node.left,  prefix + "0", codes)
    generate_codes(node.right, prefix + "1", codes)
    return codes


def huffman_encode_channel(channel_flat: np.ndarray):
    """
    Huffman-encode a flattened 1-D array of pixel values.

    Steps
    -----
    1. Count pixel value frequencies.
    2. Build Huffman tree and derive code table.
    3. Concatenate codewords for every pixel -> one long bit-string.

    Parameters
    ----------
    channel_flat : np.ndarray  shape=(N,), dtype=uint8

    Returns
    -------
    bit_string   : str          - Huffman-encoded binary string
    codes        : dict         - {pixel_value: codeword}
    root         : HuffmanNode  - tree root (needed for decoding)
    """
    freq_table = Counter(channel_flat.tolist())
    root       = build_huffman_tree(freq_table)
    codes      = generate_codes(root)

    bit_string = "".join(codes[px] for px in channel_flat.tolist())
    return bit_string, codes, root


def huffman_decode_channel(bit_string: str,
                           root: HuffmanNode,
                           total_pixels: int) -> np.ndarray:
    """
    Decode a Huffman bit-string back to the original pixel sequence.

    Walk the tree bit-by-bit: '0' -> go left, '1' -> go right.
    When a leaf is reached, record its symbol and restart from root.

    Parameters
    ----------
    bit_string    : str          - Huffman-encoded binary string
    root          : HuffmanNode  - tree root
    total_pixels  : int          - expected number of output pixels

    Returns
    -------
    np.ndarray  shape=(total_pixels,), dtype=uint8
    """
    decoded = []
    node    = root

    for bit in bit_string:
        node = node.left if bit == "0" else node.right

        if node.symbol is not None:       # reached a leaf
            decoded.append(node.symbol)
            node = root                   # restart from root

        if len(decoded) == total_pixels:  # stop early if complete
            break

    return np.array(decoded, dtype=np.uint8)


def compress_image_huffman(img_bgr: np.ndarray):
    """
    Apply Huffman coding independently to each colour channel (B, G, R).

    Because Huffman is LOSSLESS, the decoded image is byte-for-byte
    identical to the input.

    Parameters
    ----------
    img_bgr : np.ndarray  shape=(H, W, 3), dtype=uint8 - original BGR image

    Returns
    -------
    reconstructed_bgr : np.ndarray
        Decoded image (identical to input for lossless Huffman).
    total_original_bits : int
        Raw pixel bits = H * W * 3 channels * 8 bits/byte.
    total_encoded_bits : int
        Actual bits produced by Huffman encoding across all channels.
    per_channel_info : list of dict
        Per-channel statistics: {'channel', 'unique_symbols', 'avg_code_len'}.
    """
    H, W, C       = img_bgr.shape
    recon_channels = []
    total_encoded_bits   = 0
    total_original_bits  = H * W * C * 8
    per_channel_info     = []

    channel_names = ["Blue", "Green", "Red"]

    for c in range(C):
        channel_flat = img_bgr[:, :, c].flatten()   # shape = (H*W,)

        # Encode
        bit_string, codes, root = huffman_encode_channel(channel_flat)

        # Decode -> reconstruct (lossless: decoded == original)
        decoded_flat = huffman_decode_channel(bit_string, root, len(channel_flat))

        recon_channels.append(decoded_flat.reshape(H, W))
        total_encoded_bits += len(bit_string)

        # Per-channel stats
        avg_code_len = (len(bit_string) / len(channel_flat)) if channel_flat.size else 0
        per_channel_info.append({
            "channel"        : channel_names[c],
            "unique_symbols" : len(codes),
            "avg_code_len"   : avg_code_len,
        })

    reconstructed_bgr = np.stack(recon_channels, axis=2).astype(np.uint8)
    return reconstructed_bgr, total_original_bits, total_encoded_bits, per_channel_info


# =============================================================================
#  METRIC HELPERS
# =============================================================================

def calc_file_size_kb(path: str) -> float:
    """Return on-disk file size in kilobytes."""
    return os.path.getsize(path) / 1024.0


def calc_compressed_size_kb(encoded_bits: int) -> float:
    """
    Convert a Huffman bit count to an equivalent kilobyte value.
    We use ceiling division (bits -> bytes) then divide by 1024.
    """
    return math.ceil(encoded_bits / 8) / 1024.0


def calc_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error (lower is better; 0 = lossless)."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def calc_psnr(original: np.ndarray,
              reconstructed: np.ndarray,
              max_pixel: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB (higher is better).
    Returns float('inf') when images are identical (MSE == 0).
    """
    mse = calc_mse(original, reconstructed)
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(max_pixel / math.sqrt(mse))


def calc_compression_ratio(original_bits: int, encoded_bits: int) -> float:
    """
    Compression Ratio = original bits / Huffman-encoded bits.
    A ratio > 1.0 means the encoded stream is smaller than raw pixels.
    """
    if encoded_bits == 0:
        return 1.0
    return original_bits / encoded_bits


# =============================================================================
#  API CLASSIFICATION
# =============================================================================

def classify_image(client: Client, image_path: str) -> str:
    """
    Send an image to the Hugging Face Gradio Space and return the
    predicted class label as a string.

    The /predict endpoint may return a dict {"label": "...", ...}
    or a plain string; both are handled gracefully.
    """
    try:
        result = client.predict(
            handle_file(image_path),
            api_name="/predict"
        )
        if isinstance(result, dict):
            return result.get("label", str(result))
        return str(result)
    except Exception as exc:
        return f"API Error: {exc}"


# =============================================================================
#  PRETTY TABLE PRINTER
# =============================================================================

def print_table(title: str, headers: list, rows: list, col_widths: list):
    """Print a fixed-width ASCII table with a title."""
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)) + " |"
    print(f"\n{'-' * len(sep)}")
    print(f"  {title}")
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |"
        print(line)
    print(sep)


# =============================================================================
#  MAIN PROCESSING LOOP
# =============================================================================

def main():
    print("=" * 70)
    print("  Multimedia Project - Huffman Coding & HuggingFace Classification")
    print("=" * 70)
    print(f"\n  API Endpoint     : {HF_API_URL}")
    print(f"  Base Directory   : {BASE_DIR}")
    print(f"  Compression      : Huffman Coding (lossless, per BGR channel)")
    print(f"  Expected MSE     : 0.0  (lossless method)")
    print(f"  Expected PSNR    : inf  (no reconstruction error)\n")

    # --- Connect to Hugging Face Space ---------------------------------------
    print("  Connecting to Hugging Face Space API ...", end=" ", flush=True)
    try:
        client = Client(HF_API_URL)
        print("OK [connected]")
    except Exception as exc:
        print(f"FAILED\n  Error: {exc}")
        print("  Continuing without API - classification will show N/A.")
        client = None

    # --- Collect per-image results -------------------------------------------
    all_results   = []
    class_summary = {}   # class_name -> list of result dicts

    for class_name in CLASSES:
        class_dir = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"\n  [WARN] Directory not found: {class_dir} - skipping.")
            continue

        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            and "_reconstructed" not in f   # skip previously saved outputs
        ])

        if not image_files:
            print(f"\n  [WARN] No images found in {class_dir} - skipping.")
            continue

        print(f"\n  -- Processing class: {class_name} ({len(image_files)} image(s)) --")
        class_metrics = []

        for fname in image_files:
            img_path = os.path.join(class_dir, fname)
            print(f"\n    [{fname}]")

            # 1. Read original image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print("      [WARN] Could not read image - skipping.")
                continue

            # 2. Original file size
            orig_size_kb = calc_file_size_kb(img_path)
            print(f"      Original file size       : {orig_size_kb:.2f} KB")

            # 3. Huffman encode + decode (lossless round-trip)
            print("      Applying Huffman coding  ...", end=" ", flush=True)
            recon_bgr, orig_bits, enc_bits, ch_info = compress_image_huffman(img_bgr)
            print("done [OK]")

            # 4. Compressed size (in KB, from bit count)
            comp_size_kb   = calc_compressed_size_kb(enc_bits)
            comp_ratio     = calc_compression_ratio(orig_bits, enc_bits)

            # 5. Quality metrics
            mse_val  = calc_mse(img_bgr, recon_bgr)       # will be 0.0
            psnr_val = calc_psnr(img_bgr, recon_bgr)      # will be inf

            print(f"      Huffman compressed size  : {comp_size_kb:.2f} KB")
            print(f"      Compression Ratio        : {comp_ratio:.4f}  (>1 = smaller)")
            print(f"      MSE                      : {mse_val:.4f}  (0 = lossless)")
            psnr_display = "inf" if math.isinf(psnr_val) else f"{psnr_val:.2f}"
            print(f"      PSNR                     : {psnr_display} dB")

            # 6. Per-channel Huffman statistics
            for ch in ch_info:
                print(f"        {ch['channel']:5s} channel: "
                      f"{ch['unique_symbols']:3d} unique symbols, "
                      f"avg code length = {ch['avg_code_len']:.4f} bits/pixel")

            # 7. HuggingFace API Classification
            if client is not None:
                print("      Sending to API ...", end=" ", flush=True)
                prediction = classify_image(client, img_path)
                time.sleep(0.5)
                print(f"-> {prediction}")
            else:
                prediction = "N/A (API offline)"
                print(f"      API Classification       : {prediction}")

            record = {
                "class"      : class_name,
                "file"       : fname,
                "orig_kb"    : orig_size_kb,
                "comp_kb"    : comp_size_kb,
                "ratio"      : comp_ratio,
                "mse"        : mse_val,
                "psnr"       : psnr_val,        # float('inf') for lossless
                "prediction" : prediction,
                "ch_info"    : ch_info,
            }
            all_results.append(record)
            class_metrics.append(record)

        class_summary[class_name] = class_metrics

    if not all_results:
        print("\n  [ERROR] No images were processed. Check your directory layout.")
        return

    # --- Save reconstructed images -------------------------------------------
    print("\n  Saving reconstructed images ...")
    for rec in all_results:
        img_path = os.path.join(BASE_DIR, rec["class"], rec["file"])
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            continue
        recon_bgr, _, _, _ = compress_image_huffman(img_bgr)
        stem, ext  = os.path.splitext(rec["file"])
        out_name   = f"{stem}_reconstructed{ext}"
        out_path   = os.path.join(BASE_DIR, rec["class"], out_name)
        cv2.imwrite(out_path, recon_bgr)
        print(f"    Saved -> {out_path}")

    # =========================================================================
    #  TABLE 1 - Quality Evaluation
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Helper: average a numeric key over a list, handle inf
    def avg(lst, key):
        vals = [x[key] for x in lst]
        finite_vals = [v for v in vals if not math.isinf(v)]
        if not finite_vals:
            return float("inf")
        return sum(finite_vals) / len(finite_vals)

    def fmt_psnr(v):
        return "inf" if math.isinf(v) else f"{v:.2f}"

    # Per-image detail
    detail_headers = ["Class", "File", "Orig(KB)", "Huffman(KB)", "Ratio", "MSE", "PSNR(dB)", "Prediction"]
    detail_widths  = [10, 22, 10, 12, 8, 8, 10, 30]
    detail_rows    = []
    for r in all_results:
        detail_rows.append([
            r["class"],
            r["file"],
            f"{r['orig_kb']:.2f}",
            f"{r['comp_kb']:.2f}",
            f"{r['ratio']:.4f}",
            f"{r['mse']:.4f}",
            fmt_psnr(r["psnr"]),
            r["prediction"],
        ])
    print_table("Per-Image Detail", detail_headers, detail_rows, detail_widths)

    # Per-class + overall averages
    summary_headers = ["Class", "Avg Orig(KB)", "Avg Huffman(KB)", "Avg Ratio", "Avg MSE", "Avg PSNR(dB)"]
    summary_widths  = [12, 14, 17, 11, 10, 14]
    summary_rows    = []

    for cname, metrics in class_summary.items():
        summary_rows.append([
            cname,
            f"{avg(metrics, 'orig_kb'):.2f}",
            f"{avg(metrics, 'comp_kb'):.2f}",
            f"{avg(metrics, 'ratio'):.4f}",
            f"{avg(metrics, 'mse'):.4f}",
            fmt_psnr(avg(metrics, "psnr")),
        ])

    summary_rows.append([
        "OVERALL",
        f"{avg(all_results, 'orig_kb'):.2f}",
        f"{avg(all_results, 'comp_kb'):.2f}",
        f"{avg(all_results, 'ratio'):.4f}",
        f"{avg(all_results, 'mse'):.4f}",
        fmt_psnr(avg(all_results, "psnr")),
    ])

    print_table(
        "TABLE 1 - Quality Evaluation (averages per class + overall)",
        summary_headers,
        summary_rows,
        summary_widths,
    )

    # =========================================================================
    #  TABLE 2 - Comparison: My Work vs. Research Paper
    # =========================================================================
    # Replace "fill manually" cells with values from your reference paper.

    ov_orig  = avg(all_results, "orig_kb")
    ov_comp  = avg(all_results, "comp_kb")
    ov_ratio = avg(all_results, "ratio")
    ov_mse   = avg(all_results, "mse")
    ov_psnr  = avg(all_results, "psnr")

    comp_headers = ["Metric", "My Work", "Research Paper", "Notes"]
    comp_widths  = [26, 18, 20, 38]
    comp_rows    = [
        ["Avg Original Size (KB)",       f"{ov_orig:.2f}",         "<- fill manually", "Raw file on disk"],
        ["Avg Compressed Size (KB)",     f"{ov_comp:.2f}",         "<- fill manually", "Huffman bit-stream"],
        ["Avg Compression Ratio",        f"{ov_ratio:.4f}",        "<- fill manually", "Orig bits / Huffman bits"],
        ["Avg MSE",                      f"{ov_mse:.4f}",          "<- fill manually", "0 = perfect (lossless)"],
        ["Avg PSNR (dB)",                fmt_psnr(ov_psnr),        "<- fill manually", "inf = perfect (lossless)"],
        ["Compression Type",             "Lossless",               "<- fill manually", "Huffman is entropy coding"],
        ["Compression Method",           "Huffman Coding",         "<- fill manually", "Min-heap tree, per channel"],
        ["Channels Encoded",             "B, G, R (independently)","<- fill manually", "8-bit per channel"],
        ["Classifier",                   "HuggingFace API",        "<- fill manually", "m-taha6-monkeypox"],
    ]
    print_table(
        "TABLE 2 - Comparison: My Work  vs.  Research Paper",
        comp_headers,
        comp_rows,
        comp_widths,
    )

    # --- JSON export ---------------------------------------------------------
    # Convert inf to string for JSON serialisation
    export = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "psnr"}
        row["psnr"] = "inf" if math.isinf(r["psnr"]) else r["psnr"]
        export.append(row)

    out_json = os.path.join(BASE_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2, ensure_ascii=False)
    print(f"\n  Full results exported -> {out_json}")

    print("\n" + "=" * 70)
    print("  Done!  Reconstructed images saved alongside originals.")
    print("=" * 70 + "\n")


# --- Entry Point -------------------------------------------------------------
if __name__ == "__main__":
    main()
