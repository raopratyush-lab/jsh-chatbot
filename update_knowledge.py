#!/usr/bin/env python3
"""
Knowledge base updater for JSH Chatbot.
Drop MP4s, PDFs, or .txt files into the MP4/ folder and run this script.
It transcribes/extracts content, uses Claude to pull out Dr. Hiremath's
positions, and appends them to context/knowledge.md.
"""

import os
import sys
import json
import subprocess

# Auto-load API key from Netlify if not already in environment
if not os.environ.get("ANTHROPIC_API_KEY"):
    result = subprocess.run(
        ["netlify", "env:get", "ANTHROPIC_API_KEY"],
        capture_output=True, text=True, cwd=os.path.dirname(__file__)
    )
    if result.returncode == 0 and result.stdout.strip():
        os.environ["ANTHROPIC_API_KEY"] = result.stdout.strip()

import anthropic

MEDIA_DIR = os.path.join(os.path.dirname(__file__), "MP4")
KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "context", "knowledge.md")
PROCESSED_LOG = os.path.join(MEDIA_DIR, ".processed")

SUPPORTED = {
    "video": [".mp4", ".mov", ".m4v", ".mkv", ".webm"],
    "audio": [".mp3", ".m4a", ".wav", ".aac"],
    "text":  [".txt", ".md"],
    "pdf":   [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
}

client = anthropic.Anthropic()


def load_processed():
    if os.path.exists(PROCESSED_LOG):
        return set(open(PROCESSED_LOG).read().splitlines())
    return set()


def mark_processed(filename):
    with open(PROCESSED_LOG, "a") as f:
        f.write(filename + "\n")


def extract_audio(video_path):
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    print(f"  Extracting audio from {os.path.basename(video_path)}...")
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-y", audio_path],
        capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return audio_path


def transcribe(audio_path):
    print(f"  Transcribing {os.path.basename(audio_path)}...")
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except ImportError:
        # Fall back to mlx_whisper on Apple Silicon
        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(audio_path, path_or_hf_repo="mlx-community/whisper-base")
            return result["text"]
        except ImportError:
            raise RuntimeError(
                "No transcription library found.\n"
                "Install one:\n"
                "  pip3 install openai-whisper\n"
                "  or: pip3 install mlx-whisper (Apple Silicon)"
            )


def ocr_image(image_path):
    print(f"  Running OCR on {os.path.basename(image_path)}...")
    try:
        from PIL import Image
        import pytesseract
        Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check
        img = Image.open(image_path)
        # Resize very large images to speed up OCR
        max_dim = 4000
        if max(img.width, img.height) > max_dim:
            ratio = max_dim / max(img.width, img.height)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        # Convert to RGB if needed
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        text = pytesseract.image_to_string(img, lang='eng')
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    except ImportError:
        raise RuntimeError("Install Pillow and pytesseract: pip3 install pillow pytesseract")


def extract_pdf(pdf_path):
    print(f"  Extracting text from {os.path.basename(pdf_path)}...")
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        text = ""
        for page in pdf:
            text += page.get_textpage().get_text_range() + "\n"
        return text
    except ImportError:
        raise RuntimeError("pypdfium2 not installed. Run: pip3 install pypdfium2")


def get_raw_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    transcript_path = filepath.rsplit(".", 1)[0] + ".transcript.txt"

    # Return cached transcript if it exists
    if os.path.exists(transcript_path):
        print(f"  Using cached transcript: {os.path.basename(transcript_path)}")
        return open(transcript_path, encoding='utf-8', errors='ignore').read()

    if ext in SUPPORTED["video"]:
        audio = extract_audio(filepath)
        text = transcribe(audio)
        os.remove(audio)
    elif ext in SUPPORTED["audio"]:
        text = transcribe(filepath)
    elif ext in SUPPORTED["image"]:
        text = ocr_image(filepath)
    elif ext in SUPPORTED["pdf"]:
        text = extract_pdf(filepath)
    elif ext in SUPPORTED["text"]:
        return open(filepath).read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Save transcript for future reference
    with open(transcript_path, "w", encoding='utf-8', errors='ignore') as f:
        f.write(text)
    print(f"  Transcript saved: {os.path.basename(transcript_path)}")
    return text


def extract_knowledge(raw_text, filename):
    print(f"  Extracting Dr. Hiremath's positions with Claude...")
    existing = open(KNOWLEDGE_FILE).read()

    prompt = f"""You are helping build a knowledge base for an AI chatbot that speaks as Dr. Jagdish Hiremath, an interventional cardiologist based in Pune, India.

Below is a transcript or document from Dr. Hiremath (source: {filename}).

Your job:
1. Extract ONLY statements, positions, recommendations, and quotes that are clearly from Dr. Hiremath
2. IMPORTANT: Extract ONLY English text — ignore any Marathi, Hindi, or other non-English content entirely
3. Ignore host questions, filler words, unrelated conversation, and advertisements
4. Organise them under appropriate section headings (e.g. DIET, EXERCISE, CHOLESTEROL, HEART ATTACK PREVENTION, MEDICATIONS, LIFESTYLE, etc.)
5. Use the same bullet-point format as the existing knowledge base
6. Do NOT duplicate content already in the knowledge base
7. If Dr. Hiremath uses a memorable phrase or quote, preserve it in quotes
8. If the source is mostly non-English with no meaningful English content, return: SKIP

Existing knowledge base (do not duplicate):
---
{existing[:3000]}
---

Transcript/document:
---
{raw_text[:8000]}
---

Return ONLY the new knowledge to add, formatted as sections with bullet points. No preamble."""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def append_to_knowledge(new_content, source_filename):
    with open(KNOWLEDGE_FILE, "a") as f:
        f.write(f"\n\n---\n<!-- Source: {source_filename} -->\n")
        f.write(new_content)
    print(f"  ✓ Appended to knowledge.md")


def suggest_questions(new_content, filename):
    print(f"  Generating suggested questions...")
    prompt = f"""Based on the following new knowledge extracted from "{filename}", generate 8–10 suggested questions that users might ask the Dr. Jagdish Hiremath chatbot.

Make them natural, conversational, and specific to the content — not generic.

New knowledge:
---
{new_content}
---

Return a numbered list of questions only. No preamble."""

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def main():
    processed = load_processed()
    files = [
        f for f in os.listdir(MEDIA_DIR)
        if not f.startswith(".") and f not in processed
        and os.path.splitext(f)[1].lower() in
            SUPPORTED["video"] + SUPPORTED["audio"] + SUPPORTED["text"] + SUPPORTED["pdf"] + SUPPORTED["image"]
    ]

    if not files:
        print("No new files to process in MP4/")
        return

    print(f"Found {len(files)} new file(s) to process\n")

    for filename in files:
        filepath = os.path.join(MEDIA_DIR, filename)
        print(f"Processing: {filename}")
        try:
            raw_text = get_raw_text(filepath)
            if len(raw_text.strip()) < 50:
                print(f"  ⚠ Very little text extracted — skipping")
                continue
            new_knowledge = extract_knowledge(raw_text, filename)
            if new_knowledge.strip() == 'SKIP':
                print(f"  ⚠ No English content found — skipping")
                mark_processed(filename)
                continue
            append_to_knowledge(new_knowledge, filename)
            mark_processed(filename)

            questions = suggest_questions(new_knowledge, filename)
            print(f"\n  💬 Suggested questions from this source:")
            print("  " + "\n  ".join(questions.splitlines()))
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    print("\nDone! Review context/knowledge.md then run:")
    print('  git add context/knowledge.md && git commit -m "update knowledge base" && git push')


if __name__ == "__main__":
    main()
