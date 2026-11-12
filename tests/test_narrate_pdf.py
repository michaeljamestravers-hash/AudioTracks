import base64
import uuid

import pytest
from temporalio.api.enums.v1 import IndexedValueType
from temporalio.api.operatorservice.v1 import AddSearchAttributesRequest
from temporalio.testing import WorkflowEnvironment

import mistralai.workflows as workflows
from mistralai.workflows.testing import create_test_worker
from worker.workflows.narrate_pdf import (
    DriveCandidate,
    GmailCandidate,
    NarratePdfInput,
    NarratePdfWorkflow,
    SearchPlan,
    Selection,
    _drive_query,
    _is_pdf_attachment,
    _tool_json,
)

TASK_QUEUE = "test-task-queue"


def _fake_plan(prompt: str) -> SearchPlan:
    return SearchPlan(gmail_query="invoice laklaw", drive_terms=["invoice", "laklaw"])


@workflows.activity(name="narrate-pdf-plan-search")
async def fake_plan(prompt: str) -> SearchPlan:
    return _fake_plan(prompt)


def make_fake_activities(
    drive_files,
    gmail_attachments,
    selection,
    pdf_b64=None,
    ocr_text="Invoice text.",
    audio_b64="YXVkaW8=",
    saved_file=None,
):
    drive_files = list(drive_files)
    gmail_attachments = list(gmail_attachments)
    calls = []

    @workflows.activity(name="narrate-pdf-search-drive")
    async def fake_search_drive(terms: list) -> list:
        calls.append(("search_drive", terms))
        return [DriveCandidate.model_validate(f) for f in drive_files]

    @workflows.activity(name="narrate-pdf-search-gmail")
    async def fake_search_gmail(gmail_query: str) -> list:
        calls.append(("search_gmail", gmail_query))
        return [GmailCandidate.model_validate(f) for f in gmail_attachments]

    @workflows.activity(name="narrate-pdf-pick")
    async def fake_pick(prompt: str, drive_candidates: list, gmail_candidates: list) -> Selection:
        calls.append(("pick", prompt, drive_candidates, gmail_candidates))
        return selection

    @workflows.activity(name="narrate-pdf-download")
    async def fake_download(file_id: str) -> str:
        calls.append(("download", file_id))
        return pdf_b64 or base64.b64encode(b"%PDF-fake").decode()

    @workflows.activity(name="narrate-pdf-ocr")
    async def fake_ocr(pdf_b64: str, filename: str) -> str:
        calls.append(("ocr", filename))
        return ocr_text

    @workflows.activity(name="narrate-pdf-tts")
    async def fake_tts(text: str, voice_id: str, max_chars: int) -> str:
        calls.append(("tts", text, voice_id, max_chars))
        return audio_b64

    @workflows.activity(name="narrate-pdf-save-to-drive")
    async def fake_save(filename: str, audio_b64: str) -> dict:
        calls.append(("save", filename))
        return saved_file or {"id": "saved-1", "view_url": "https://example.com/a.mp3"}

    return [
        fake_plan,
        fake_search_drive,
        fake_search_gmail,
        fake_pick,
        fake_download,
        fake_ocr,
        fake_tts,
        fake_save,
    ], calls


@pytest.fixture
async def env():
    environment = await WorkflowEnvironment.start_local()
    await environment.client.operator_service.add_search_attributes(
        AddSearchAttributesRequest(
            namespace=environment.client.namespace,
            search_attributes={"OtelTraceId": IndexedValueType.INDEXED_VALUE_TYPE_KEYWORD},
        )
    )
    yield environment
    await environment.shutdown()


async def run_workflow(env, activities, prompt="the law office invoice"):
    async with create_test_worker(
        env,
        [NarratePdfWorkflow],
        activities=activities,
        task_queue=TASK_QUEUE,
    ):
        handle = await env.client.start_workflow(
            "narrate-pdf",
            args=[NarratePdfInput(prompt=prompt)],
            id=f"test-{uuid.uuid4()}",
            task_queue=TASK_QUEUE,
        )
        result = await handle.result()
    return result


def test_drive_query_escapes_and_filters():
    query = _drive_query("alice's invoice")
    assert "mimeType = 'application/pdf'" in query
    assert "title contains 'alice\\'s invoice'" in query


def test_tool_json_extracts_mcp_text_content():
    envelope = {"content": [{"type": "text", "text": '{\"files\": [{\"id\": \"f1\"}]}'}], "metadata": {}}
    assert _tool_json(envelope) == {"files": [{"id": "f1"}]}
    assert _tool_json({"files": []}) == {"files": []}
    assert _tool_json({"content": []}) is None
    assert _tool_json({"content": [{"type": "text", "text": "not json"}]}) is None


def test_is_pdf_attachment():
    assert _is_pdf_attachment({"filename": "a.PDF", "mimeType": ""})
    assert _is_pdf_attachment({"filename": "a.txt", "mimeType": "application/pdf"})
    assert not _is_pdf_attachment({"filename": "a.txt", "mimeType": "text/plain"})


async def test_drive_pdf_is_narrated_and_saved(env):
    selection = Selection(
        source="gdrive", file_id="file-1", title="invoice.pdf", reason="best"
    )
    activities, calls = make_fake_activities(
        drive_files=[
            {
                "file_id": "file-1",
                "title": "invoice.pdf",
                "modified_time": "2026-11-01T00:00:00Z",
            }
        ],
        gmail_attachments=[],
        selection=selection,
        ocr_text="Invoice for October.",
    )
    result = await run_workflow(env, activities)

    assert result["status"] == "narrated"
    assert result["pdf_title"] == "invoice.pdf"
    assert result["source"] == "gdrive"
    assert result["drive_file_id"] == "saved-1"
    assert result["view_url"] == "https://example.com/a.mp3"

    names = [c[0] for c in calls]
    assert names == ["search_drive", "search_gmail", "pick", "download", "ocr", "tts", "save"]
    save_call = calls[-1]
    assert save_call[1] == "invoice.pdf (narration).mp3"


async def test_gmail_only_attachment_reports_fallback(env):
    gmail_candidate = {
        "message_id": "msg-1",
        "thread_id": "t-1",
        "attachment_id": "att-1",
        "filename": "invoice.pdf",
        "subject": "Invoice",
        "sender": "accounting@example.com",
        "date": "2026-11-11",
    }
    selection = Selection(
        source="gmail",
        message_id="msg-1",
        attachment_id="att-1",
        title="invoice.pdf",
        reason="only gmail",
    )
    activities, calls = make_fake_activities(
        drive_files=[],
        gmail_attachments=[gmail_candidate],
        selection=selection,
    )
    result = await run_workflow(env, activities)

    assert result["status"] == "gmail_attachment_found"
    assert result["source"] == "gmail"
    assert result["pdf_title"] == "invoice.pdf"
    assert "Save the attachment" in result["message"]

    names = [c[0] for c in calls]
    assert "download" not in names
    assert "tts" not in names


async def test_no_match_reports_not_found(env):
    selection = Selection(source="none", reason="nothing found")
    activities, _ = make_fake_activities(
        drive_files=[], gmail_attachments=[], selection=selection
    )
    result = await run_workflow(env, activities)

    assert result["status"] == "not_found"
    assert result["source"] is None
