from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from rag_platform.core.errors import AppError, PayloadTooLargeError, UnsupportedMediaTypeError
from rag_platform.infrastructure.storage import LocalFileStorage, StorageError
from rag_platform.services.documents import DocumentService, sanitise_filename
from tests.conftest import make_settings
from tests.pdf_factory import make_pdf


async def chunks_of(data: bytes, size: int = 1000) -> AsyncIterator[bytes]:
    for start in range(0, len(data), size):
        yield data[start : start + size]


@pytest.fixture
async def storage(tmp_path: Path) -> LocalFileStorage:
    storage = LocalFileStorage(tmp_path)
    await storage.ensure_ready()
    return storage


def service(storage: LocalFileStorage, **overrides: Any) -> DocumentService:
    # stage() never touches the database, so no session is needed.
    return DocumentService(None, storage, make_settings(**overrides))  # type: ignore[arg-type]


def incoming_files(tmp_path: Path) -> list[Path]:
    return list((tmp_path / ".incoming").iterdir())


async def test_stage_hashes_and_measures_pdf(storage: LocalFileStorage, tmp_path: Path) -> None:
    data = make_pdf(["hello"])
    staged = await service(storage).stage(chunks_of(data))
    assert staged.size_bytes == len(data)
    assert len(staged.sha256) == 64
    assert staged.temp_path.read_bytes() == data


async def test_stage_rejects_oversized_file(storage: LocalFileStorage, tmp_path: Path) -> None:
    data = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 1)
    with pytest.raises(PayloadTooLargeError):
        await service(storage, max_upload_size_mb=1).stage(chunks_of(data, 64 * 1024))
    assert incoming_files(tmp_path) == []


async def test_stage_rejects_non_pdf(storage: LocalFileStorage, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        await service(storage).stage(chunks_of(b"<html><body>not a pdf</body></html>"))
    assert incoming_files(tmp_path) == []


async def test_stage_rejects_empty_file(storage: LocalFileStorage, tmp_path: Path) -> None:
    with pytest.raises(AppError) as info:
        await service(storage).stage(chunks_of(b""))
    assert info.value.code == "empty_file"
    assert incoming_files(tmp_path) == []


async def test_pdf_signature_split_across_chunks_is_detected(storage: LocalFileStorage) -> None:
    staged = await service(storage).stage(chunks_of(make_pdf(["x"]), size=2))
    assert staged.size_bytes > 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\secret.pdf", "secret.pdf"),
        ('bad<>:"|?*name.pdf', "bad_______name.pdf"),
        ("line\nbreak.pdf", "line_break.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("...", "document.pdf"),
    ],
)
def test_sanitise_filename(raw: str | None, expected: str) -> None:
    assert sanitise_filename(raw) == expected


def test_sanitise_filename_truncates_but_keeps_extension() -> None:
    name = sanitise_filename("a" * 400 + ".pdf")
    assert len(name) <= 255
    assert name.endswith(".pdf")


def test_storage_refuses_keys_outside_root(storage: LocalFileStorage) -> None:
    with pytest.raises(StorageError):
        storage.path_for("../outside.pdf")
