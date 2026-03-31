"""
RenAIssance - Handwritten Historical Document OCR Pipeline
GSoC 2026 | HumanAI Foundation

Applicant: Sarthak Sharma
Test: Test II - Text recognition of handwritten sources using LLM/VLM pipeline

Run: python renaissance_ocr.py
"""

import subprocess, sys

REQUIRED = ['pymupdf==1.23.8', 'pillow', 'opencv-python',
            'requests', 'jiwer', 'python-docx', 'rich', 'numpy']

for pkg in REQUIRED:
    name = pkg.split('==')[0]
    try:
        __import__({'pymupdf':'fitz','pillow':'PIL','opencv-python':'cv2',
                    'python-docx':'docx'}.get(name, name.replace('-','_')))
    except ImportError:
        print(f'Installing {pkg}...')
        subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])

import os, json, time, base64, io, re, requests
from docx import Document
from jiwer import cer, wer
import fitz
import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()

# ── CONFIG ────────────────────────────────────────────────────────────────────

PDF_DIR           = r"C:\renaissance\Handwriting-20260331T000942Z-3-001\Handwriting"
TRANSCRIPTION_DIR = r"C:\renaissance\Handwriting-20260331T000856Z-3-001\Handwriting"
OUTPUT_DIR        = r"C:\renaissance\output"

MODEL_BACKEND   = 'gemini'       # 'ollama' | 'gemini' | 'openai'
OLLAMA_BASE_URL = 'http://localhost:11434'
VISION_MODEL    = 'llava'
TEXT_MODEL      = 'llama3.1:8b'
GEMINI_API_KEY  = 'AIzaSyAjTGdGqTOoN0OOoCYEyOmPjkhJPk9Skhc'
OPENAI_API_KEY  = ''

PAGES_PER_DOC   = 1      # 2 pages per doc to avoid timeouts
DPI             = 72    # lower DPI = smaller image = faster llava
TIMEOUT         = 600    # 10 minutes per request

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── STEP 1: LOAD DATASET ─────────────────────────────────────────────────────

def clean_name(name):
    """Strip URL-encoded chars and punctuation for fuzzy matching."""
    name = name.replace('&#x3a_', '').replace('&#x3a;', '').replace(':_', '')
    name = name.replace('_transcription', '').replace('.pdf','').replace('.docx','')
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def load_dataset():
    pdf_files  = sorted([f for f in os.listdir(PDF_DIR)  if f.endswith('.pdf')])
    docx_files = sorted([f for f in os.listdir(TRANSCRIPTION_DIR) if f.endswith('.docx')])

    console.print(f"\n[bold blue]Found {len(pdf_files)} PDFs and {len(docx_files)} transcriptions[/bold blue]")

    documents = []
    for pdf in pdf_files:
        pdf_key = clean_name(pdf)
        match = None
        best  = 0
        for docx in docx_files:
            docx_key = clean_name(docx)
            common = sum(1 for a, b in zip(pdf_key, docx_key) if a == b)
            if common > best:
                best  = common
                match = docx
        if best < 8:
            match = None
        documents.append({'pdf': pdf, 'transcription': match})
        status = f"[green]{match}[/green]" if match else "[red]NO MATCH[/red]"
        console.print(f"  {pdf[:55]:55} -> {status}")

    return documents


# ── STEP 2: LOAD GROUND TRUTH ─────────────────────────────────────────────────

def load_ground_truth(documents):
    ground_truth = {}
    for doc in documents:
        if not doc['transcription']:
            continue
        path = os.path.join(TRANSCRIPTION_DIR, doc['transcription'])
        d    = Document(path)
        text = '\n'.join([p.text for p in d.paragraphs if p.text.strip()])
        ground_truth[doc['pdf']] = text
        console.print(f"  [green]Loaded[/green] {doc['transcription']} ({len(text)} chars)")
    return ground_truth


# ── STEP 3: PREPROCESSING ─────────────────────────────────────────────────────

def pdf_to_images(pdf_path, dpi=120, max_pages=2):
    """Convert PDF pages to PIL images using pymupdf."""
    doc    = fitz.open(pdf_path)
    images = []
    for i in range(min(len(doc), max_pages)):
        page = doc[i]
        mat  = fitz.Matrix(dpi/72, dpi/72)
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def preprocess_image(pil_img, max_width=1000):
    """
    Preprocessing pipeline for historical handwritten documents:
    - Resize to max 1000px width
    - Convert to greyscale
    - CLAHE contrast enhancement for faded/uneven ink
    - Keep as greyscale (NOT binarised) — binarisation loses detail in handwriting
    """
    w, h = pil_img.size
    if w > max_width:
        ratio   = max_width / w
        pil_img = pil_img.resize((max_width, int(h * ratio)), Image.LANCZOS)

    img      = np.array(pil_img)
    grey     = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grey)
    # return greyscale — not binarised, preserves ink gradients for VLM
    return Image.fromarray(enhanced).convert('RGB')

def image_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── STEP 4: LLM/VLM BACKENDS ─────────────────────────────────────────────────

VISION_PROMPT = (
    "You are an expert paleographer specialising in early modern Spanish manuscripts "
    "from the 16th to 19th centuries. "
    "Transcribe the handwritten text visible in this document image. "
    "1) Transcribe exactly as written including archaic spellings. "
    "2) Do not modernise historical spelling. "
    "3) Mark illegible words with [illegible]. "
    "4) Ignore marginalia, main body text only. "
    "5) Also correct any obvious misread characters in your output. "
    "Output only the transcription, nothing else."
)

CORRECTION_PROMPT = (
    "You are an expert in early modern Spanish manuscripts (siglos XVI-XIX). "
    "Review and correct only clear OCR errors — misread characters, broken words, "
    "obvious nonsense sequences. "
    "Do NOT modernise archaic spellings (vn=un, q=que, dho=dicho are correct). "
    "Do NOT add punctuation. Return only the corrected text.\n\nRAW TEXT:\n"
)

def transcribe_ollama(image_b64):
    r = requests.post(
        f'{OLLAMA_BASE_URL}/api/generate',
        json={'model': VISION_MODEL, 'prompt': VISION_PROMPT,
              'images': [image_b64], 'stream': False},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()['response']

def correct_ollama(raw_text):
    r = requests.post(
        f'{OLLAMA_BASE_URL}/api/generate',
        json={'model': TEXT_MODEL,
              'prompt': CORRECTION_PROMPT + raw_text, 'stream': False},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()['response']

def transcribe_gemini(image_b64):
    from google import genai
    from google.genai import types
    client   = genai.Client(api_key=GEMINI_API_KEY)
    img_data = base64.b64decode(image_b64)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            VISION_PROMPT,
            types.Part.from_bytes(data=img_data, mime_type='image/png')
        ]
    )
    return response.text

def transcribe_openai(image_b64):
    r = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {OPENAI_API_KEY}',
                 'Content-Type': 'application/json'},
        json={'model': 'gpt-4-vision-preview', 'max_tokens': 2000,
              'messages': [{'role': 'user', 'content': [
                  {'type': 'text', 'text': VISION_PROMPT},
                  {'type': 'image_url', 'image_url': {
                      'url': f'data:image/png;base64,{image_b64}'}}
              ]}]},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

def transcribe_image(image_b64):
    if MODEL_BACKEND == 'ollama': return transcribe_ollama(image_b64)
    if MODEL_BACKEND == 'gemini': return transcribe_gemini(image_b64)
    if MODEL_BACKEND == 'openai': return transcribe_openai(image_b64)
    raise ValueError(f'Unknown backend: {MODEL_BACKEND}')

def correct_transcription(raw):
    if MODEL_BACKEND == 'ollama':
        return correct_ollama(raw)
    return raw


# ── STEP 5: EVALUATION ───────────────────────────────────────────────────────

def normalise(text):
    return re.sub(r'\s+', ' ', text.lower().strip())

def evaluate(all_results, ground_truth):
    rows = []
    for pdf_name, pages in all_results.items():
        if pdf_name not in ground_truth:
            console.print(f"[yellow]No ground truth for {pdf_name[:40]}[/yellow]")
            continue
        raw  = normalise(' '.join(p['raw']       for p in pages))
        corr = normalise(' '.join(p['corrected'] for p in pages))
        ref  = normalise(ground_truth[pdf_name])

        rc = cer(ref, raw);  cc = cer(ref, corr)
        rw = wer(ref, raw);  cw = wer(ref, corr)
        imp = ((rc - cc) / rc * 100) if rc > 0 else 0

        rows.append({'document': pdf_name, 'pages': len(pages),
                     'raw_cer': rc, 'corr_cer': cc,
                     'raw_wer': rw, 'corr_wer': cw, 'imp': imp})
    return rows

def print_table(rows):
    t = Table(title="Evaluation Results")
    t.add_column("Document",   style="cyan", max_width=30)
    t.add_column("Pg",         justify="right")
    t.add_column("Raw CER",    justify="right")
    t.add_column("Corr CER",   justify="right", style="green")
    t.add_column("Improve",    justify="right", style="yellow")
    t.add_column("Corr WER",   justify="right")
    for r in rows:
        t.add_row(r['document'][:30], str(r['pages']),
                  f"{r['raw_cer']*100:.1f}%",  f"{r['corr_cer']*100:.1f}%",
                  f"{r['imp']:.1f}%",           f"{r['corr_wer']*100:.1f}%")
    if rows:
        ac = sum(r['corr_cer'] for r in rows) / len(rows)
        aw = sum(r['corr_wer'] for r in rows) / len(rows)
        ai = sum(r['imp']      for r in rows) / len(rows)
        t.add_row("[bold]AVERAGE[/bold]", "",
                  "", f"[bold]{ac*100:.1f}%[/bold]",
                  f"[bold]{ai:.1f}%[/bold]", f"[bold]{aw*100:.1f}%[/bold]")
    console.print(t)


# ── STEP 6: SAVE ─────────────────────────────────────────────────────────────

def save_results(all_results, evaluation):
    with open(os.path.join(OUTPUT_DIR, 'transcription_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    for pdf_name, pages in all_results.items():
        out = os.path.join(OUTPUT_DIR, pdf_name.replace('.pdf', '_transcription.txt'))
        with open(out, 'w', encoding='utf-8') as f:
            f.write(f"Document: {pdf_name}\nBackend: {MODEL_BACKEND}/{VISION_MODEL}\n{'='*60}\n\n")
            for p in pages:
                f.write(f"--- Page {p['page']} ---\n{p['corrected']}\n\n")
        console.print(f"  [green]Saved[/green] {os.path.basename(out)}")
    console.print(f"\n[bold green]Results saved to {OUTPUT_DIR}[/bold green]")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold cyan]RenAIssance OCR Pipeline[/bold cyan]")
    console.print(f"Backend: [yellow]{MODEL_BACKEND}[/yellow] | "
                  f"Vision: [yellow]{VISION_MODEL}[/yellow] | "
                  f"Text: [yellow]{TEXT_MODEL}[/yellow] | "
                  f"Timeout: [yellow]{TIMEOUT}s[/yellow]\n")

    if MODEL_BACKEND == 'ollama':
        try:
            r      = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
            models = [m['name'] for m in r.json()['models']]
            console.print(f"[green]Ollama running.[/green] Models: {models}\n")
        except Exception as e:
            console.print(f"[red]Ollama not reachable: {e}[/red]")
            return

    console.rule("Step 1 — Load Dataset")
    documents = load_dataset()

    console.rule("Step 2 — Load Ground Truth")
    ground_truth = load_ground_truth(documents)

    console.rule("Step 3 — OCR Pipeline")
    all_results = {}

    for doc_info in documents:
        pdf_name = doc_info['pdf']
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        console.print(f"\n[bold]Processing:[/bold] {pdf_name}")

        images = pdf_to_images(pdf_path, dpi=DPI, max_pages=PAGES_PER_DOC)
        # force resize to max 800px wide before sending to llava
        resized = []
        for img in images:
            w, h = img.size
            if w > 800:
                img = img.resize((800, int(h * 800 / w)), Image.LANCZOS)
            resized.append(img)
        images = resized
        console.print(f"  {len(images)} pages resized to {images[0].size if images else 'n/a'}")

        doc_results = []
        for i, img in enumerate(images):
            console.print(f"  Page {i+1}/{len(images)}...")

            processed = preprocess_image(img)
            img_b64   = image_to_base64(processed)
            processed.save(os.path.join(OUTPUT_DIR, f"{pdf_name[:15]}_p{i+1}.png"))

            console.print(f"    [cyan]Stage 1[/cyan] transcribing ({VISION_MODEL})...")
            t0 = time.time()
            try:
                raw = transcribe_image(img_b64)
                t1  = time.time() - t0
                console.print(f"    {t1:.1f}s | {len(raw)} chars")
                console.print(f"    Preview: {raw[:100]}...")
            except Exception as e:
                t1 = time.time() - t0
                console.print(f"    [red]Failed after {t1:.1f}s: {e}[/red]")
                raw = "[transcription failed — timeout]"

            # Stage 2: LLM correction using llama3.1 with historical Spanish context
            console.print(f"    [cyan]Stage 2[/cyan] correcting ({TEXT_MODEL})...")
            t2 = time.time()
            try:
                corrected = correct_ollama(raw)
                t2 = time.time() - t2
                console.print(f"    {t2:.1f}s")
            except Exception as e:
                console.print(f"    [yellow]Correction skipped: {e}[/yellow]")
                corrected = raw
                t2 = 0.0

            # use raw gemini output as final — llm correction degrades quality
            # on archaic spanish by modernising authentic historical spelling
            doc_results.append({
                'page': i+1, 'raw': raw, 'corrected': raw,
                't_vision': round(t1, 2), 't_correct': round(t2, 2),
                'llm_corrected': corrected  # kept for comparison
            })

        all_results[pdf_name] = doc_results
        console.print(f"  [green]Done[/green]")

    console.rule("Step 4 — Evaluation")
    evaluation = evaluate(all_results, ground_truth)
    print_table(evaluation)

    console.rule("Step 5 — Save")
    save_results(all_results, evaluation)


if __name__ == '__main__':
    main()
