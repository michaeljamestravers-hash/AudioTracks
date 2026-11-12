"""Narrate a PDF chosen with a natural-language prompt.

Searches Google Drive and Gmail for the described PDF, extracts its text with
Mistral OCR, narrates it with Voxtral text-to-speech, and saves the audio to Drive.
"""

from __future__ import annotations

import base64
import re
from typing import Any, List, Optional

import pydantic

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

SEARCH_MODEL = "mistral-small-latest"
OCR_MODEL = "mistral-ocr-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_VOICE = "gb_jane_neutral"
MAX_DRIVE_CANDIDATES = 10
MAX_GMAIL_THREADS = 10

drive_connector = connector("google_drive_mcp")
gmail_connector = connector("gmail")


def _mistral_client():
    from mistralai.client import Mistral

    return Mistral()


class NarratePdfInput(pydantic.BaseModel):
    prompt: str = pydantic.Field(
        description="Describe the PDF to narrate, e.g. 'the law office invoice from last week'."
    )
    voice_id: str = pydantic.Field(
        default=DEFAULT_VOICE,
        description="Voxtral preset voice slug for the narration.",
    )
    max_chars: int = pydantic.Field(
        default=20000,
        description="Maximum number of characters to narrate, taken from the start of the document.",
    )


class NarratePdfOutput(pydantic.BaseModel):
    status: str
    pdf_title: Optional[str] = None
    source: Optional[str] = None
    drive_file_id: Optional[str] = None
    view_url: Optional[str] = None
    message: str


class SearchPlan(pydantic.BaseModel):
    gmail_query: str = pydantic.Field(
        description="Gmail search query for messages with the PDF attached, using Gmail syntax."
    )
    drive_terms: List[str] = pydantic.Field(
        description="Up to 4 short search terms likely to appear in the PDF's Drive title or content."
    )


class DriveCandidate(pydantic.BaseModel):
    file_id: str
    title: str
    modified_time: Optional[str] = None


class GmailCandidate(pydantic.BaseModel):
    message_id: str
    thread_id: str
    attachment_id: str
    filename: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    date: Optional[str] = None


class Selection(pydantic.BaseModel):
    source: str = pydantic.Field(description="'gdrive', 'gmail', or 'none'.")
    file_id: Optional[str] = None
    message_id: Optional[str] = None
    attachment_id: Optional[str] = None
    title: Optional[str] = None
    reason: Optional[str] = None


@workflows.activity(name="narrate-pdf-plan-search")
async def plan_search(prompt: str) -> SearchPlan:
    client = _mistral_client()
    response = await client.chat.parse_async(
        model=SEARCH_MODEL,
        response_format=SearchPlan,
        messages=[
            {
                "role": "system",
                "content": (
                    "You turn a user's description of a PDF document into search queries. "
                    "For Gmail, produce one query using Gmail search syntax that finds messages "
                    "with that PDF attached; include 'has:attachment' and 'filename:pdf' plus "
                    "the most distinctive terms (sender, subject words, date hints). "
                    "For Drive, produce up to 4 short terms that would appear in the file's "
                    "title or content. Reply only with the structured plan."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return SearchPlan.model_validate(response.choices[0].message.parsed)


def _drive_query(term: str) -> str:
    escaped = term.replace("'", "\\'")
    return (
        f"mimeType = 'application/pdf' and "
        f"(title contains '{escaped}' or fullText contains '{escaped}')"
    )


@workflows.activity(name="narrate-pdf-search-drive")
async def search_drive(
    terms: List[str],
    drive: ToolCallClient = Depends(drive_connector),
) -> List[DriveCandidate]:
    seen: dict[str, DriveCandidate] = {}
    for term in terms[:4]:
        result = await drive.call_tool(
            tool_name="search_files",
            arguments={"query": _drive_query(term), "page_size": MAX_DRIVE_CANDIDATES},
        )
        for file in (result or {}).get("files") or []:
            file_id = file.get("id")
            if not file_id or file_id in seen:
                continue
            seen[file_id] = DriveCandidate(
                file_id=file_id,
                title=file.get("title") or "",
                modified_time=file.get("modified_time"),
            )
            if len(seen) >= MAX_DRIVE_CANDIDATES:
                return list(seen.values())
    return list(seen.values())


def _is_pdf_attachment(attachment: dict[str, Any]) -> bool:
    filename = (attachment.get("filename") or "").lower()
    mime = attachment.get("mimeType") or ""
    return mime == "application/pdf" or filename.endswith(".pdf")


@workflows.activity(name="narrate-pdf-search-gmail")
async def search_gmail(
    gmail_query: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> List[GmailCandidate]:
    result = await gmail.call_tool(
        tool_name="search_threads",
        arguments={"query": f"has:attachment filename:pdf {gmail_query}", "pageSize": MAX_GMAIL_THREADS},
    )
    candidates: List[GmailCandidate] = []
    for thread in (result or {}).get("threads") or []:
        thread_id = thread.get("id") or ""
        for message in thread.get("messages") or []:
            for attachment in message.get("attachments") or []:
                if not _is_pdf_attachment(attachment):
                    continue
                candidates.append(
                    GmailCandidate(
                        message_id=message.get("id") or "",
                        thread_id=thread_id,
                        attachment_id=attachment.get("id") or "",
                        filename=attachment.get("filename") or "",
                        subject=message.get("subject"),
                        sender=message.get("sender"),
                        date=message.get("date"),
                    )
                )
    return candidates


@workflows.activity(name="narrate-pdf-pick")
async def pick_pdf(
    prompt: str,
    drive_candidates: List[DriveCandidate],
    gmail_candidates: List[GmailCandidate],
) -> Selection:
    if not drive_candidates and not gmail_candidates:
        return Selection(source="none", reason="No PDF matched the prompt in Drive or Gmail.")
    client = _mistral_client()
    response = await client.chat.parse_async(
        model=SEARCH_MODEL,
        response_format=Selection,
        messages=[
            {
                "role": "system",
                "content": (
                    "You pick which file the user means from search candidates. "
                    "Prefer Google Drive candidates, because a Drive PDF can be narrated "
                    "directly while a Gmail attachment cannot be downloaded. Pick a Gmail "
                    "candidate only when no Drive candidate plausibly matches. "
                    "For the chosen candidate, copy its identifiers exactly. "
                    "If nothing plausibly matches, use source 'none'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User prompt: {prompt}\n\n"
                    f"Drive candidates: "
                    f"{[c.model_dump() for c in drive_candidates]}\n\n"
                    f"Gmail candidates: "
                    f"{[c.model_dump() for c in gmail_candidates]}"
                ),
            },
        ],
    )
    return Selection.model_validate(response.choices[0].message.parsed)


@workflows.activity(name="narrate-pdf-download")
async def download_pdf(
    file_id: str,
    drive: ToolCallClient = Depends(drive_connector),
) -> str:
    result = await drive.call_tool(
        tool_name="download_file_content",
        arguments={"file_id": file_id},
    )
    content = (result or {}).get("content")
    if not content:
        raise workflows.ActivityError(f"Drive file {file_id} has no downloadable content.")
    return content


@workflows.activity(name="narrate-pdf-ocr")
async def ocr_pdf(pdf_b64: str, filename: str) -> str:
    from mistralai.client.models import DocumentURLChunk, File

    pdf_bytes = base64.b64decode(pdf_b64)
    client = _mistral_client()

    uploaded = client.files.upload(
        file=File(file_name=filename, content=pdf_bytes),
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id)
    ocr_response = client.ocr.process(
        model=OCR_MODEL,
        document=DocumentURLChunk(document_url=signed.url),
    )
    pages = sorted(ocr_response.pages, key=lambda page: page.index)
    return "\n\n".join((page.markdown or "").strip() for page in pages).strip()


@workflows.activity(name="narrate-pdf-tts")
async def narrate_text(text: str, voice_id: str, max_chars: int) -> str:
    trimmed = text[:max_chars].strip()
    if not trimmed:
        raise workflows.ActivityError("The PDF contains no extractable text to narrate.")
    client = _mistral_client()
    response = client.audio.speech.complete(
        model=TTS_MODEL,
        input=trimmed,
        voice_id=voice_id,
        response_format="mp3",
    )
    return response.audio_data


@workflows.activity(name="narrate-pdf-save-to-drive")
async def save_to_drive(
    filename: str,
    audio_b64: str,
    drive: ToolCallClient = Depends(drive_connector),
) -> dict[str, Any]:
    result = await drive.call_tool(
        tool_name="create_file",
        arguments={
            "title": filename,
            "base64_content": audio_b64,
            "content_mime_type": "audio/mpeg",
        },
    )
    return result or {}


@workflows.workflow.define(
    name="narrate-pdf",
    workflow_display_name="Narrate PDF",
    workflow_description=(
        "Narrate a PDF chosen with a natural-language prompt. Searches Gmail and Google "
        "Drive, reads the PDF with OCR, and saves a spoken MP3 narration to Google Drive."
    ),
    on_behalf_of=True,
)
@uses_connectors(drive_connector, gmail_connector)
class NarratePdfWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: NarratePdfInput) -> NarratePdfOutput:
        plan = await plan_search(input.prompt)
        drive_candidates = await search_drive(plan.drive_terms)
        gmail_candidates = await search_gmail(plan.gmail_query)
        selection = await pick_pdf(input.prompt, drive_candidates, gmail_candidates)

        if selection.source == "none":
            return NarratePdfOutput(
                status="not_found",
                message=(
                    "No PDF matching your prompt was found in Google Drive or Gmail. "
                    f"Picker reason: {selection.reason or 'no plausible match'}."
                ),
            )

        if selection.source == "gmail":
            match = next(
                (c for c in gmail_candidates if c.message_id == selection.message_id),
                None,
            )
            subject = match.subject if match else None
            sender = match.sender if match else None
            filename = match.filename if match else selection.title
            return NarratePdfOutput(
                status="gmail_attachment_found",
                pdf_title=filename or "PDF attachment",
                source="gmail",
                message=(
                    f"Found the PDF as a Gmail attachment"
                    f"{' from ' + sender if sender else ''}"
                    f"{' about ' + repr(subject) if subject else ''}, but Gmail attachments "
                    "cannot be downloaded through the Gmail connector. Save the attachment "
                    "to Google Drive and run this workflow again to have it narrated."
                ),
            )

        file_id = selection.file_id
        if not file_id:
            return NarratePdfOutput(
                status="not_found",
                message="The selected PDF could not be identified. Try a more specific prompt.",
            )

        pdf_b64 = await download_pdf(file_id)
        title = selection.title or f"drive-file-{file_id}.pdf"
        text = await ocr_pdf(pdf_b64, re.sub(r"[^\w .-]", "_", title))
        audio_b64 = await narrate_text(text, input.voice_id, input.max_chars)
        saved = await save_to_drive(f"{title} (narration).mp3", audio_b64)

        return NarratePdfOutput(
            status="narrated",
            pdf_title=title,
            source="gdrive",
            drive_file_id=saved.get("id"),
            view_url=saved.get("view_url"),
            message="Narration saved to Google Drive.",
        )
