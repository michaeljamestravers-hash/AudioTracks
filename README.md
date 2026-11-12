# AudioTracks

A Mistral Workflow that narrates a PDF you choose with a natural-language prompt.

Give it a prompt like *"the law office invoice from last week"* or *"the presentation
about the Paris trip"*. The workflow searches your Google Drive and Gmail, picks the
best-matching PDF, extracts its text with Mistral OCR, generates narration audio with
Voxtral text-to-speech, and saves the MP3 back to your Google Drive.

## How it works

1. **Plan** — a Mistral model turns your prompt into Gmail and Drive search queries.
2. **Search** — Google Drive is searched for PDFs (`google_drive_mcp` connector),
   and Gmail for messages with PDF attachments (`gmail` connector).
3. **Pick** — the model ranks the candidates and picks the best match.
4. **Narrate** — the PDF text is extracted with `mistral-ocr-latest` and spoken with
   `voxtral-mini-tts-3-0`.
5. **Deliver** — the MP3 is saved to your Drive as `<pdf title> (narration).mp3`.

If the PDF exists only as a Gmail attachment, the Gmail connector cannot download
attachment bytes, so the workflow reports the exact email and attachment and asks you
to save it to Drive and re-run.

The workflow runs on behalf of the user who starts it, so it uses **your own**
Gmail and Drive credentials.

## Input

```json
{
  "prompt": "the law office invoice from last week",
  "voice_id": "gb_jane_neutral",
  "max_chars": 20000
}
```

- `prompt` (required): describe the PDF in plain words.
- `voice_id` (optional): any preset Voxtral voice slug, e.g. `gb_jane_neutral`.
- `max_chars` (optional): cap on narration length; the PDF is narrated from the start.

## Output

```json
{
  "status": "narrated",
  "pdf_title": "Travers invoice_492.pdf",
  "source": "gdrive",
  "drive_file_id": "…",
  "view_url": "https://drive.google.com/…",
  "message": "Narration saved to Google Drive."
}
```

`status` is one of `narrated`, `gmail_attachment_found`, `not_found`.

## Run locally

```bash
uv sync --extra dev
cp .env.example .env   # then set MISTRAL_API_KEY
source .env && uv run python -m worker.entrypoints.worker
```

## Test

```bash
uv run pytest
```
