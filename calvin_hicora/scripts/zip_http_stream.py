#!/usr/bin/env python3
"""HTTP Range helpers and sequential ZIP streaming for CALVIN archives.

Freiburg serves ``Accept-Ranges: bytes``. We never materialize the full zip on
disk or in a giant ``BytesIO``. Two access modes:

1. ``HttpRangeFile`` — seekable file-like used with ``zipfile.ZipFile`` to read
   the central directory from the end and pull sparse metadata members.
2. ``ResumableHttpStream`` + ``iter_zip_local_members`` — one sequential body
   download (with byte-offset resume) that yields each local member as it
   arrives, so millions of ``episode_*.npz`` files do not become millions of
   HTTP round-trips.
"""

from __future__ import annotations

import hashlib
import io
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Callable, Iterator, Optional


LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
CENTRAL_DIRECTORY_SIGNATURE = 0x02014B50
EOCD_SIGNATURE = 0x06054B50
ZIP64_EOCD_LOCATOR_SIGNATURE = 0x07064B50

STORED = 0
DEFLATED = 8


def _urlopen(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


def head_content_length(url: str, timeout: float = 60.0) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with _urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        accept = (response.headers.get("Accept-Ranges") or "").lower()
        if length is None:
            raise RuntimeError(f"HEAD {url} did not return Content-Length")
        size = int(length)
    if accept == "bytes":
        return size
    # Some servers omit Accept-Ranges on HEAD; probe with an actual Range GET.
    probe = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with _urlopen(probe, timeout=timeout) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 206:
                raise RuntimeError(
                    f"Server for {url} does not support HTTP Range "
                    f"(HEAD Accept-Ranges={accept!r}, probe status={status}); "
                    "refuse non-resumable download of a 100+ GiB zip"
                )
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                size = int(content_range.rsplit("/", 1)[1])
            response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Range probe failed for {url}: {exc}") from exc
    print(f"[HEAD] Range probe ok (Accept-Ranges header was {accept!r})", flush=True)
    return size


def fetch_text(url: str, timeout: float = 60.0) -> str:
    with _urlopen(urllib.request.Request(url), timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_official_sha256(checksum_text: str, archive_name: str) -> str:
    for line in checksum_text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        digest, name = parts[0], parts[-1].lstrip("*")
        if name == archive_name:
            return digest.lower()
    raise KeyError(f"{archive_name} absent from official sha256sum.txt")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote object via HTTP Range requests.

    Used only for cheap random access (EOCD / central directory / a handful of
    metadata members). Do not iterate millions of episode files through this
    path — use ``ResumableHttpStream`` instead.
    """

    def __init__(
        self,
        url: str,
        size: int,
        *,
        timeout: float = 120.0,
        max_retries: int = 8,
        retry_sleep_seconds: float = 2.0,
        on_request: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        super().__init__()
        self.url = url
        self.size = int(size)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.on_request = on_request
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if self._pos < 0:
            raise ValueError("negative seek")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self._pos
        if size <= 0 or self._pos >= self.size:
            return b""
        start = self._pos
        end = min(self.size, self._pos + size) - 1
        data = self._range_get(start, end)
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        data = self.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        return n

    def _range_get(self, start: int, end: int) -> bytes:
        if self.on_request is not None:
            self.on_request(start, end)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                self.url,
                headers={"Range": f"bytes={start}-{end}"},
            )
            try:
                with _urlopen(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", None) or response.getcode()
                    if status not in (200, 206):
                        raise RuntimeError(f"Range GET status={status}")
                    data = response.read()
                if not data and start <= end:
                    raise RuntimeError(f"Empty Range response for bytes={start}-{end}")
                return data
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                last_error = exc
                sleep = self.retry_sleep_seconds * (2**attempt)
                print(
                    f"[HttpRangeFile] retry {attempt + 1}/{self.max_retries} "
                    f"bytes={start}-{end}: {exc}; sleep {sleep:.1f}s",
                    flush=True,
                )
                time.sleep(sleep)
        raise RuntimeError(f"Range GET failed for bytes={start}-{end}: {last_error}")


@dataclass
class ZipLocalMember:
    filename: str
    compress_type: int
    compress_size: int
    file_size: int
    header_offset: int
    payload_offset: int
    data: bytes


class ResumableHttpStream:
    """Sequential HTTP body reader with byte-offset resume and running SHA-256."""

    def __init__(
        self,
        url: str,
        size: int,
        *,
        start_offset: int = 0,
        timeout: float = 120.0,
        max_retries: int = 12,
        retry_sleep_seconds: float = 2.0,
        chunk_size: int = 8 * 1024 * 1024,
        hash_stream: bool = True,
    ) -> None:
        self.url = url
        self.size = int(size)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.chunk_size = chunk_size
        self.offset = int(start_offset)
        self.hash_stream = hash_stream
        self._digest = hashlib.sha256() if hash_stream and start_offset == 0 else None
        self._hash_complete = False
        self._response = None
        self._buffer = bytearray()
        if start_offset < 0 or start_offset > self.size:
            raise ValueError(f"start_offset {start_offset} out of range for size {size}")
        if start_offset > 0 and hash_stream:
            # Cannot continue a SHA-256 mid-stream without saved hash state.
            # Provenance then relies on official sha256sum.txt.
            print(
                "[ResumableHttpStream] resume mid-archive: stream SHA-256 disabled; "
                "use official sha256sum.txt for provenance",
                flush=True,
            )
        self._open_response()

    @property
    def bytes_consumed(self) -> int:
        return self.offset

    def hexdigest(self) -> str | None:
        if self._digest is None:
            return None
        if not self._hash_complete and self.offset != self.size:
            return None
        return self._digest.hexdigest()

    def close(self) -> None:
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None

    def _open_response(self) -> None:
        self.close()
        headers = {}
        if self.offset > 0:
            headers["Range"] = f"bytes={self.offset}-"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(self.url, headers=headers)
            try:
                response = _urlopen(request, timeout=self.timeout)
                status = getattr(response, "status", None) or response.getcode()
                if self.offset == 0 and status not in (200, 206):
                    raise RuntimeError(f"GET status={status}")
                if self.offset > 0 and status != 206:
                    raise RuntimeError(
                        f"Resume requires HTTP 206, got {status}; "
                        "server may have dropped Range support"
                    )
                self._response = response
                print(
                    f"[ResumableHttpStream] open offset={self.offset}/{self.size} "
                    f"status={status}",
                    flush=True,
                )
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                last_error = exc
                sleep = self.retry_sleep_seconds * (2**attempt)
                print(
                    f"[ResumableHttpStream] open retry {attempt + 1}/{self.max_retries}: "
                    f"{exc}; sleep {sleep:.1f}s",
                    flush=True,
                )
                time.sleep(sleep)
        raise RuntimeError(f"Failed to open HTTP stream at offset {self.offset}: {last_error}")

    def read(self, n: int) -> bytes:
        if n <= 0:
            return b""
        out = bytearray()
        while len(out) < n and (self._buffer or self.offset < self.size):
            need = n - len(out)
            if self._buffer:
                take = min(need, len(self._buffer))
                out.extend(self._buffer[:take])
                del self._buffer[:take]
                continue
            raw = self._read_chunk(min(self.chunk_size, self.size - self.offset))
            if not raw:
                break
            if len(raw) <= need:
                out.extend(raw)
            else:
                out.extend(raw[:need])
                self._buffer.extend(raw[need:])
        return bytes(out)

    def _read_chunk(self, n: int) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self._response is None:
                    self._open_response()
                assert self._response is not None
                data = self._response.read(n)
                if not data and self.offset < self.size:
                    raise RuntimeError("unexpected EOF from HTTP body")
                if data:
                    self.offset += len(data)
                    if self._digest is not None:
                        self._digest.update(data)
                        if self.offset >= self.size:
                            self._hash_complete = True
                return data
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                last_error = exc
                sleep = self.retry_sleep_seconds * (2**attempt)
                print(
                    f"[ResumableHttpStream] read retry {attempt + 1}/{self.max_retries} "
                    f"at offset={self.offset}: {exc}; sleep {sleep:.1f}s",
                    flush=True,
                )
                time.sleep(sleep)
                self._open_response()
        raise RuntimeError(f"HTTP read failed at offset {self.offset}: {last_error}")

    def discard(self, n: int) -> None:
        remaining = n
        while remaining > 0:
            chunk = self.read(min(self.chunk_size, remaining))
            if not chunk:
                raise EOFError(f"discard hit EOF with {remaining} bytes left")
            remaining -= len(chunk)


def _read_exact(stream: BinaryIO | ResumableHttpStream, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


def _decompress_member(compress_type: int, payload: bytes, file_size: int) -> bytes:
    if compress_type == STORED:
        data = payload
    elif compress_type == DEFLATED:
        data = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"unsupported ZIP compress_type={compress_type}")
    if file_size and len(data) != file_size:
        raise ValueError(f"uncompressed size mismatch: {len(data)} != {file_size}")
    return data


def iter_zip_local_members(
    stream: BinaryIO | ResumableHttpStream,
    *,
    start_offset: int = 0,
    should_materialize: Callable[[str], bool] | None = None,
) -> Iterator[ZipLocalMember]:
    """Parse ZIP local file headers from a sequential stream.

    Members for which ``should_materialize`` returns False have their compressed
    payload discarded without inflate (RGB-heavy episode files that we still
    must skip after a resume, directories, etc.). Central directory signatures
    stop iteration.
    """
    offset = int(start_offset)
    while True:
        header_offset = offset
        sig_bytes = stream.read(4)
        if not sig_bytes:
            return
        if len(sig_bytes) < 4:
            raise EOFError("truncated ZIP signature")
        (signature,) = struct.unpack("<I", sig_bytes)
        if signature == CENTRAL_DIRECTORY_SIGNATURE or signature == EOCD_SIGNATURE:
            print(
                f"[iter_zip_local_members] reached central directory at offset={header_offset}",
                flush=True,
            )
            return
        if signature == ZIP64_EOCD_LOCATOR_SIGNATURE:
            print(
                f"[iter_zip_local_members] reached ZIP64 locator at offset={header_offset}",
                flush=True,
            )
            return
        if signature != LOCAL_FILE_HEADER_SIGNATURE:
            raise ValueError(f"bad local header signature {hex(signature)} at {header_offset}")

        header = _read_exact(stream, 26)
        offset += 4 + 26
        (
            _ver,
            flag,
            compress_type,
            _time,
            _date,
            _crc,
            compress_size,
            file_size,
            name_len,
            extra_len,
        ) = struct.unpack("<HHHHHIIIHH", header)
        if flag & 0x08:
            raise ValueError(
                f"data-descriptor ZIP member at offset {header_offset}; "
                "CALVIN archives are expected to store sizes in the local header"
            )
        filename = _read_exact(stream, name_len).decode("utf-8", errors="replace")
        extra = _read_exact(stream, extra_len)
        offset += name_len + extra_len
        # ZIP64 extra field may override 0xFFFFFFFF sizes.
        if compress_size == 0xFFFFFFFF or file_size == 0xFFFFFFFF:
            compress_size, file_size = _parse_zip64_sizes(extra, compress_size, file_size)
        payload_offset = offset
        materialize = True if should_materialize is None else should_materialize(filename)
        if compress_size < 0:
            raise ValueError(f"negative compress_size for {filename}")
        if materialize and compress_size > 0:
            payload = _read_exact(stream, compress_size)
            data = _decompress_member(compress_type, payload, file_size)
        elif materialize:
            data = b""
        else:
            if hasattr(stream, "discard"):
                stream.discard(compress_size)  # type: ignore[union-attr]
            else:
                _read_exact(stream, compress_size)
            data = b""
        offset += compress_size
        yield ZipLocalMember(
            filename=filename,
            compress_type=compress_type,
            compress_size=compress_size,
            file_size=file_size,
            header_offset=header_offset,
            payload_offset=payload_offset,
            data=data,
        )


def _parse_zip64_sizes(extra: bytes, compress_size: int, file_size: int) -> tuple[int, int]:
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        chunk = extra[pos : pos + data_size]
        pos += data_size
        if header_id != 0x0001:
            continue
        cursor = 0
        if file_size == 0xFFFFFFFF:
            (file_size,) = struct.unpack_from("<Q", chunk, cursor)
            cursor += 8
        if compress_size == 0xFFFFFFFF:
            (compress_size,) = struct.unpack_from("<Q", chunk, cursor)
        return compress_size, file_size
    raise ValueError("ZIP64 sizes required but extra field 0x0001 missing")
