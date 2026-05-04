# Skin Disease Classification with Huffman Coding Compression

> **Multimedia Systems Project — Mohamed Taha, 2026**

A Python pipeline that classifies skin disease images (**Measles** vs **Monkeypox**) using a deep-learning model hosted on Hugging Face, and evaluates **Huffman Coding** as a lossless image compression method. Quality metrics (Compression Ratio, MSE, PSNR) are calculated for every image and summarised in two report-ready tables.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Dependencies & Installation](#3-dependencies--installation)
4. [How to Run](#4-how-to-run)
5. [Huffman Coding — Algorithm Explained](#5-huffman-coding--algorithm-explained)
6. [Metrics Explained](#6-metrics-explained)
7. [Results](#7-results)
8. [Output Files](#8-output-files)
9. [Understanding the Tables](#9-understanding-the-tables)
10. [FAQ / Common Questions](#10-faq--common-questions)

---

## 1. Project Overview

| Component | Details |
|-----------|---------|
| **Task** | Binary image classification: Measles vs Monkeypox |
| **Classifier** | Custom CNN model deployed at [m-taha6-monkeypox.hf.space](https://m-taha6-monkeypox.hf.space/) |
| **Compression Method** | Huffman Coding (lossless entropy coding) |
| **Channels** | Blue, Green, Red — encoded independently |
| **Metrics** | Original Size, Compressed Size, Compression Ratio, MSE, PSNR |

The core idea is to demonstrate that images can be **losslessly compressed** using Huffman Coding — meaning the reconstructed image is **pixel-for-pixel identical** to the original — while also reporting how efficiently the algorithm encodes the data compared to raw pixels.

---

## 2. Directory Structure

```
Muiltmedia_project/
│
├── Measles/
│   ├── measles10.png                   ← original test image
│   ├── measles11.png                   ← original test image
│   ├── measles10_reconstructed.png     ← Huffman decoded (identical to original)
│   └── measles11_reconstructed.png     ← Huffman decoded (identical to original)
│
├── Monkeypox/
│   ├── monkeypox179.png
│   ├── monkeypox241.png
│   ├── monkeypox179_reconstructed.png
│   └── monkeypox241_reconstructed.png
│
├── multimedia_analysis.py              ← main script (this project)
├── results.json                        ← full per-image results (auto-generated)
└── README.md                           ← this file
```

---

## 3. Dependencies & Installation

Python **3.9+** is required. Install all dependencies with one command:

```bash
python -m pip install gradio_client opencv-python numpy
```

| Package | Purpose |
|---------|---------|
| `gradio_client` | Calls the Hugging Face Spaces API to classify images |
| `opencv-python` | Reads / writes images, BGR colour handling |
| `numpy` | Array operations for pixel manipulation |
| `heapq` *(stdlib)* | Min-heap used in Huffman tree construction |
| `collections` *(stdlib)* | `Counter` for frequency table |

> **No external Huffman library is used.** The entire algorithm is implemented from scratch inside `multimedia_analysis.py`.

---

## 4. How to Run

```bash
cd e:\Muiltmedia_project
python multimedia_analysis.py
```

The script will:
1. Connect to the Hugging Face API
2. Process every image in `Measles/` and `Monkeypox/`
3. Apply Huffman compression + decompression per image
4. Print per-image statistics and both result tables
5. Save `*_reconstructed.png` files and `results.json`

**Expected runtime:** ~30–60 seconds (dominated by API round-trips).

---

## 5. Huffman Coding — Algorithm Explained

Huffman Coding is a **lossless entropy coding** technique. It assigns shorter binary codewords to more-frequent pixel values and longer codewords to rarer ones, so the total bit count is minimised.

### Step-by-step pipeline

```
Original Image (BGR)
        │
        ▼
┌───────────────────────────────────────────────┐
│  For each colour channel (Blue, Green, Red):  │
│                                               │
│  1. Count pixel value frequencies (0–255)     │
│     e.g. {128: 5000, 200: 3000, 45: 100 ...} │
│                                               │
│  2. Build a min-heap of leaf nodes            │
│     (one node per unique pixel value)         │
│                                               │
│  3. Merge the two lowest-frequency nodes      │
│     repeatedly until one root remains         │
│     → this is the Huffman Tree                │
│                                               │
│  4. Traverse tree to assign binary codes      │
│     Left edge = '0',  Right edge = '1'        │
│     Frequent pixels → short codes (e.g. "10") │
│     Rare    pixels → long  codes (e.g. "110001") │
│                                               │
│  5. Encode: replace each pixel with its code  │
│     → produces a long binary bit-string       │
│                                               │
│  6. Decode: walk the tree bit-by-bit          │
│     '0' → go left, '1' → go right            │
│     Reach a leaf → output its pixel value     │
│     → reconstructed channel (lossless)        │
└───────────────────────────────────────────────┘
        │
        ▼
Reconstructed Image (BGR) — pixel-for-pixel identical to original
```

### Why is MSE = 0 and PSNR = ∞?

Because Huffman is **lossless** — no information is discarded during encoding. The decoded output is mathematically identical to the input, so:

- **MSE** (Mean Squared Error) = 0 — no pixel differs
- **PSNR** (Peak Signal-to-Noise Ratio) = ∞ — no noise was introduced

This is the expected, correct result for any lossless compression method.

### Compression Ratio explained

```
Compression Ratio = Raw pixel bits / Huffman-encoded bits

Raw pixel bits = Height × Width × 3 channels × 8 bits/pixel
```

A ratio **> 1.0** means the Huffman stream uses fewer bits than the raw uncompressed pixels.

> **Note:** The "Huffman KB" column in the tables may be *larger* than the on-disk PNG file size. This is because PNG already uses DEFLATE compression internally. The comparison baseline in this project is **raw uncompressed pixel data**, which is the standard academic baseline for evaluating compression algorithms.

---

## 6. Metrics Explained

| Metric | Formula | Ideal Value | Meaning |
|--------|---------|-------------|---------|
| **Original Size (KB)** | `file size / 1024` | — | On-disk size of the original image |
| **Compressed Size (KB)** | `ceil(Huffman bits / 8) / 1024` | Smaller than original | Estimated size of the Huffman bit-stream |
| **Compression Ratio** | `raw bits / huffman bits` | > 1.0 | How many times smaller the encoded data is vs raw pixels |
| **MSE** | `mean((original − reconstructed)²)` | 0 (lossless) | Average squared pixel error |
| **PSNR (dB)** | `20 × log₁₀(255 / √MSE)` | ∞ (lossless) | Signal quality vs noise; > 30 dB is considered visually acceptable |
| **Avg Code Length** | `encoded bits / total pixels` | < 8 bits | Average bits used per pixel (theoretical entropy limit) |

---

## 7. Results

### Per-Image Results

| Class | File | Orig (KB) | Huffman (KB) | Ratio | MSE | PSNR | Prediction |
|-------|------|-----------|-------------|-------|-----|------|------------|
| Measles | measles10.png | 79.21 | 129.75 | 1.1330 | 0.0000 | inf | Measles ✓ |
| Measles | measles11.png | 85.08 | 124.99 | 1.1761 | 0.0000 | inf | Measles ✓ |
| Monkeypox | monkeypox179.png | 72.61 | 130.86 | 1.1233 | 0.0000 | inf | Monkeypox ✓ |
| Monkeypox | monkeypox241.png | 84.20 | 138.55 | 1.0610 | 0.0000 | inf | Monkeypox ✓ |

### TABLE 1 — Quality Evaluation (Averages)

| Class | Avg Orig (KB) | Avg Huffman (KB) | Avg Ratio | Avg MSE | Avg PSNR |
|-------|-------------|-----------------|-----------|---------|----------|
| Measles | 82.14 | 127.37 | 1.1545 | 0.0000 | inf |
| Monkeypox | 78.41 | 134.71 | 1.0922 | 0.0000 | inf |
| **OVERALL** | **80.27** | **131.04** | **1.1233** | **0.0000** | **inf** |

### Per-Channel Huffman Statistics

| Class | File | Channel | Unique Symbols | Avg Code Length (bits/px) |
|-------|------|---------|---------------|--------------------------|
| Measles | measles10.png | Blue | 192 | 7.1230 |
| Measles | measles10.png | Green | 179 | 6.9706 |
| Measles | measles10.png | Red | 200 | 7.0898 |
| Measles | measles11.png | Blue | 195 | 7.1012 |
| Measles | measles11.png | Green | 180 | 6.8122 |
| Measles | measles11.png | Red | 149 | 6.4927 |
| Monkeypox | monkeypox179.png | Blue | 206 | 7.0573 |
| Monkeypox | monkeypox179.png | Green | 200 | 6.9521 |
| Monkeypox | monkeypox179.png | Red | 198 | 7.3561 |
| Monkeypox | monkeypox241.png | Blue | 248 | 7.5761 |
| Monkeypox | monkeypox241.png | Green | 242 | 7.4724 |
| Monkeypox | monkeypox241.png | Red | 236 | 7.5715 |

> The theoretical maximum for 8-bit pixels is **8 bits/pixel**. Achieving ~7 bits/pixel means the Huffman tree successfully exploits the non-uniform pixel distribution to reduce entropy.

### TABLE 2 — Comparison Template (fill in Research Paper column)

| Metric | My Work | Research Paper | Notes |
|--------|---------|---------------|-------|
| Avg Original Size (KB) | 80.27 | ← fill manually | Raw file on disk |
| Avg Compressed Size (KB) | 131.04 | ← fill manually | Huffman bit-stream |
| Avg Compression Ratio | 1.1233 | ← fill manually | Orig bits / Huffman bits |
| Avg MSE | 0.0000 | ← fill manually | 0 = perfect (lossless) |
| Avg PSNR (dB) | inf | ← fill manually | inf = perfect (lossless) |
| Compression Type | Lossless | ← fill manually | Huffman is entropy coding |
| Compression Method | Huffman Coding | ← fill manually | Min-heap tree, per channel |
| Channels Encoded | B, G, R independently | ← fill manually | 8-bit per channel |
| Classifier | HuggingFace API | ← fill manually | m-taha6-monkeypox |

---

## 8. Output Files

| File | Description |
|------|-------------|
| `results.json` | Full per-image metrics in JSON format (auto-generated) |
| `Measles/*_reconstructed.png` | Huffman-decoded Measles images |
| `Monkeypox/*_reconstructed.png` | Huffman-decoded Monkeypox images |

---

## 9. Understanding the Tables

### Why is the Huffman (KB) column larger than Orig (KB)?

The on-disk PNG file is **already compressed** by the PNG encoder (which uses DEFLATE — a more advanced algorithm than plain Huffman). When we apply *standalone* Huffman coding to raw pixels and measure the resulting bit count, it will sometimes be larger than the highly-optimised PNG format. However, the ratio **is still > 1.0** because the baseline is raw *uncompressed* pixel data (H × W × 3 × 8 bits), which is the academically correct comparison.

### Why is the Classifier 100% accurate?

The test images were selected from the same dataset distribution as the training data, so the model recognises them confidently. All 4 images were correctly classified.

---

## 10. FAQ / Common Questions

**Q: Why use Huffman Coding instead of JPEG/DCT?**  
A: Huffman is a foundational algorithm in information theory and forms the entropy-coding backbone of formats like JPEG, ZIP, and PNG. Studying it in isolation demonstrates the core compression principle without lossy artefacts.

**Q: Why is MSE = 0?**  
A: Because Huffman is **lossless** — every decoded pixel value is identical to the original. This is mathematically guaranteed by the algorithm design.

**Q: Can I add more images?**  
A: Yes. Simply place `.png` / `.jpg` files in the `Measles/` or `Monkeypox/` folders and re-run the script. It automatically skips `*_reconstructed.png` files from previous runs.

**Q: Can I change the compression method back to DCT?**  
A: The previous DCT version of the script used `scipy.fftpack`. The Huffman version replaced it entirely. Both versions are fully documented in the project's conversation history.

**Q: How do I fill in the Research Paper column in Table 2?**  
A: Find the metrics reported in your reference paper (MSE, PSNR, Compression Ratio) and manually replace the `<- fill manually` placeholders directly in the console output or in your report document.

---

*Generated by `multimedia_analysis.py` — Mohamed Taha, 2026*
