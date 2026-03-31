# RenAIssance OCR Pipeline
### GSoC 2026 Evaluation Test | HumanAI Foundation

**Applicant:** Sarthak Sharma  
**Project:** End-to-end handwritten text recognition for early modern Spanish documents with LLM or Vision-Language Model pipeline creation  
**Test:** Test II — Text recognition of handwritten sources using LLM/VLM pipeline  
**GitHub:** [sarthak-here](https://github.com/sarthak-here) | **Email:** sarthak909999@gmail.com

---

## Overview

This repository contains my evaluation test submission for the RenAIssance project under HumanAI Foundation (GSoC 2026). The pipeline performs end-to-end OCR on early modern Spanish handwritten manuscripts (16th–19th century) using a two-stage LLM/VLM approach — with the VLM integrated as the primary transcription engine throughout the process, not just as a post-processing step.

---

## Dataset

5 handwritten historical Spanish documents spanning 1606–1857:

| File | Period | Description |
|---|---|---|
| AHPG-GPAH 1.1716, A.35 | 1744 | Hidalguia certificate |
| AHPG-GPAH AU61.2 | 1606 | Legal declaration |
| ES.28079.AHN INQUISICIÓN, 1667 | 1640 | Inquisition document |
| PT3279.146.342 | 1857 | Notarial record |
| Pleito entre el Marqués de Viana | ~1600 | Legal dispute |

---

## Pipeline Architecture

```
PDF Document
     ↓
Convert to image (pymupdf @ 72 DPI)
     ↓
Preprocess (CLAHE contrast enhancement + greyscale)
     ↓
Resize to 800px width (for VLM performance)
     ↓
Stage 1: Gemini 2.5 Flash (VLM) — reads handwriting directly
     ↓
Stage 2: llama3.1:8b (LLM) — applies historical Spanish context
     ↓
Evaluate with CER + WER (jiwer) vs human reference transcriptions
     ↓
Save results (JSON + .txt per document)
```

### Why LLM/VLM throughout — not just at the end

The project specification requires LLM/VLM to be used throughout the process. This pipeline uses:

- **Stage 1 — Gemini 2.5 Flash:** The VLM directly reads the handwritten image using paleographic expertise encoded in the prompt. It handles character recognition, layout interpretation, and archaic abbreviation reading in one pass.
- **Stage 2 — llama3.1:8b:** A language model applies historical Spanish linguistic context to correct residual OCR errors.

### Preprocessing decisions

**CLAHE (Contrast Limited Adaptive Histogram Equalisation)** was chosen over simple global thresholding because historical documents have highly non-uniform contrast — ink fades unevenly, staining affects some regions more than others. CLAHE adapts to local regions independently.

**Greyscale, not binarised** — early testing showed that Otsu binarisation lost too much detail in faded ink regions, causing the VLM to refuse to transcribe ("image too blurry"). Keeping greyscale preserves ink gradients that the VLM needs.

**800px max width** — the source PDFs render at 3000–5000px. Sending full-resolution images to Gemini caused timeouts. 800px preserves enough detail for character recognition while keeping API response times under 60s.

---

## Results

| Document | Period | Raw CER | Corrected CER | Notes |
|---|---|---|---|---|
| AHPG-GPAH 1.1716, A.35 | 1744 | 39.3% | 38.3% | Slight improvement |
| AHPG-GPAH AU61.2 | 1606 | 26.6% | 27.0% | Marginal degradation |
| ES.28079 Inquisición | 1640 | 38.5% | 39.0% | LLM modernised spelling |
| PT3279.146.342 | 1857 | 32.7% | 35.4% | LLM over-corrected |
| Pleito Marqués de Viana | ~1600 | 43.4% | 90.1% | LLM completely rewrote |
| **Average** | — | **35.6%** | 46.0% | **Raw Gemini better** |

### Key Finding

The generic LLM correction stage **worsened** average CER from 35.6% to 46% by modernising archaic Spanish spelling. For example, historical forms like `vn` (for `un`), `dho` (for `dicho`), and `q` with abbreviation marks were treated as OCR errors and "corrected" to modern Spanish — which is linguistically wrong and increases CER against the human reference.

**The raw Gemini 2.5 Flash output (35.6% avg CER) is the better result and is used as the final transcription.**

This finding directly motivates the core GSoC contribution: a **self-supervised DAN model fine-tuned on historical Spanish data** combined with a **domain-specific correction model** trained on period-accurate Spanish — not a generic LLM.

### Sample transcriptions (raw Gemini output)

**AHPG-GPAH AU61.2 (1606):**
```
Ena de aguirre Vesina dela Villa de Oñate y madre legitima de maria
Ana de arrequi mi Hija Legítima...
```

**PT3279.146.342 (1857):**
```
En una villa de Lizarza a nueve de Junio de mil ocho-
cientos Cincuenta y Seis, ante mi el escrivano...
```

---

## How to Improve Toward 90% Accuracy

The 35.6% CER baseline is a strong zero-shot starting point. The following improvements are planned for the GSoC project:

1. **Fine-tune DAN (Document Attention Network)** on historical Spanish handwriting — DAN processes entire pages at once, capturing layout and reading order without line segmentation. Research shows fine-tuned DAN achieves under 10% CER on comparable historical document tasks.

2. **Self-supervised pre-training** on large unlabelled historical Spanish collections (PARES, BNE Digital, Europeana) using masked character prediction before supervised fine-tuning.

3. **Domain-specific correction model** — fine-tune on CORDE (Real Academia Española historical corpus) to learn 16th–19th century Spanish conventions. Unlike generic LLMs, this model will know that `vn`, `dho`, and `q` are correct historical forms.

4. **TrOCR fine-tuning** as a complementary line-level recogniser for cleaner document sections.

5. **Confidence-based ensemble** — route ambiguous/degraded regions to Gemini, clearer regions to fine-tuned DAN/TrOCR.

6. **Historian annotation tool** — web interface for human experts to validate and correct OCR output, feeding corrections back into the training pipeline.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/sarthak-here/renaissance-gsoc2026
cd renaissance-gsoc2026

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install google-genai pymupdf==1.23.8 pillow opencv-python requests jiwer python-docx rich numpy
```

---

## Configuration

Open `renaissance_ocr.py` and set your paths and API key:

```python
# Paths — update to match your local setup
PDF_DIR           = r"path\to\handwriting\pdfs"
TRANSCRIPTION_DIR = r"path\to\transcription\docx"
OUTPUT_DIR        = r"path\to\output"

# Backend — 'gemini', 'ollama', or 'openai'
MODEL_BACKEND  = 'gemini'
GEMINI_API_KEY = 'your-key-here'  # get from aistudio.google.com
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

---

## Usage

```bash
python renaissance_ocr.py
```

The script will:
1. Load all PDFs and ground truth transcriptions
2. Convert each PDF page to image and preprocess
3. Send each image to Gemini 2.5 Flash for transcription
4. Apply LLM correction (optional)
5. Evaluate CER/WER against human references
6. Save results to the output directory

---

## File Structure

```
renaissance-gsoc2026/
├── renaissance_ocr.py       # main pipeline script
├── requirements.txt         # dependencies
└── README.md
```

Output files (generated on run):
```
output/
├── transcription_results.json       # full results per document
├── evaluation_report.json           # CER/WER metrics
├── *_transcription.txt              # readable text per document
└── *_p1.png                         # preprocessed page images
```

---

## Dependencies

```
google-genai
pymupdf==1.23.8
pillow
opencv-python
requests
jiwer
python-docx
rich
numpy
```

---

## Evaluation Metrics

**CER (Character Error Rate)** is the primary metric for historical OCR. It measures edit distance at the character level and is more appropriate than WER for archaic vocabulary — WER would penalise correctly transcribed historical words simply because they don't match modern spellings.

```
CER = (insertions + deletions + substitutions) / total reference characters
```

A CER below 10% is considered production-quality for historical document OCR. The 35.6% zero-shot baseline is a strong starting point for a model with no domain-specific training on this material.

---

## Supported Backends

| Backend | Model | Notes |
|---|---|---|
| `gemini` | gemini-2.5-flash | Recommended — best accuracy, free API |
| `ollama` | llava (local) | Fully offline, slower, lower accuracy |
| `openai` | gpt-4-vision-preview | High accuracy, paid API |

Switch backend in the config section at the top of `renaissance_ocr.py`.
