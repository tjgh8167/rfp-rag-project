import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import olefile
from pypdf import PdfReader



HWP_PARA_TEXT_TAG = 67
HWP_LIST_HEADER_TAG = 72
HWP_TABLE_TAG = 77
# 표 셀의 LIST_HEADER는 공통 헤더 8바이트 뒤에 열·행·가로병합·세로병합이 2바이트씩 이어집니다.
HWP_CELL_POSITION_OFFSET = 8
HWP_IMAGE_SUFFIXES = {'.bmp', '.emf', '.gif', '.jpeg', '.jpg', '.png', '.svg', '.tif', '.tiff', '.wmf'}
HWP_SINGLE_CONTROL_CHARS = {9, 10, 13, 24, 30, 31}
CHUNK_REQUIRED_FIELDS = ("chunk_id", "doc_id", "text", "metadata")
REQUIRED_METADATA_FIELDS = ("title", "project_name", "agency", "file_name")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict


# HWP 본문 스트림의 Section 번호를 숫자로 변환합니다.
def _section_number(stream_path: list[str]) -> int:
    return int(stream_path[-1].removeprefix("Section"))


# HWP 본문에 포함된 제어문자를 제거하고 읽을 수 있는 텍스트로 정리합니다.
def _clean_hwp_text(text: str) -> str:
    cleaned = []
    position = 0

    while position < len(text):
        code = ord(text[position])
        if code >= 32:
            cleaned.append(text[position])
            position += 1
        elif code in HWP_SINGLE_CONTROL_CHARS:
            cleaned.append("\n" if code in {10, 13} else " ")
            position += 1
        else:
            position += 8

    return "".join(cleaned).strip()


# HWP 5.x 본문 스트림의 레코드를 순서대로 읽습니다.
def _iter_hwp_body_records(hwp: olefile.OleFileIO, compressed: bool):
    section_paths = sorted(
        (
            stream_path
            for stream_path in hwp.listdir()
            if len(stream_path) == 2
            and stream_path[0] == "BodyText"
            and stream_path[1].startswith("Section")
        ),
        key=_section_number,
    )

    for stream_path in section_paths:
        section = hwp.openstream(stream_path).read()
        if compressed:
            section = zlib.decompress(section, -15)

        position = 0
        while position + 4 <= len(section):
            record_header = int.from_bytes(section[position : position + 4], "little")
            position += 4
            tag_id = record_header & 0x3FF
            level = (record_header >> 10) & 0x3FF
            record_size = (record_header >> 20) & 0xFFF

            if record_size == 0xFFF:
                if position + 4 > len(section):
                    break
                record_size = int.from_bytes(section[position : position + 4], "little")
                position += 4

            record = section[position : position + record_size]
            position += record_size
            yield tag_id, level, record


# 표 하나가 끝났을 때 본문에 남길지 결정합니다.
# 표 청크가 담당하는 표는 본문에서 빼고, 담당하지 않는 표는 글자만 본문에 넣습니다.
def _flush_hwp_table(table: dict, options: dict, paragraphs: list[str]) -> None:
    if is_hwp_content_table(
        table,
        options["min_rows"],
        options["min_columns"],
        options["min_density"],
    ):
        return
    for cell in table["cells"]:
        paragraphs.extend(cell["text"])


# HWP 5.x 파일의 압축된 본문 스트림을 열어 텍스트를 추출합니다.
# exclude_tables=True면 표 안의 글자를 건너뜁니다. 표는 extract_hwp_tables로 따로 뽑아
# 표 청크로 관리하므로, 본문에 함께 넣으면 같은 내용이 두 벌이 됩니다.
# 다만 표 청크로 가지 않는 표(1행·1열 제목 상자 등)는 본문에 남깁니다.
# 그 표에는 장·절 제목이 들어 있어 빼면 목차가 어디에도 남지 않습니다.
def _read_hwp(
    path: Path,
    exclude_tables: bool = False,
    table_options: dict | None = None,
) -> str:

    if not olefile.isOleFile(path):
        raise ValueError("HWP 5.x OLE 문서가 아닙니다.")

    paragraphs = []
    with olefile.OleFileIO(path) as hwp:
        if not hwp.exists("FileHeader") or not hwp.exists("BodyText"):
            raise ValueError("HWP FileHeader 또는 BodyText 스트림이 없습니다.")

        file_header = hwp.openstream("FileHeader").read()
        if not file_header.startswith(b"HWP Document File"):
            raise ValueError("지원하지 않는 HWP 문서입니다.")

        compressed = bool(file_header[36] & 1)
        options = table_options or {}
        open_tables: list[dict] = []
        for tag_id, level, record in _iter_hwp_body_records(hwp, compressed):
            if exclude_tables:
                # 표보다 얕은 계층이 나오면 그 표가 끝난 것이므로 결과를 판정합니다.
                while open_tables and level < open_tables[-1]["level"]:
                    _flush_hwp_table(open_tables.pop(), options, paragraphs)
                if tag_id == HWP_TABLE_TAG and len(record) >= 8:
                    open_tables.append(
                        {
                            "rows": int.from_bytes(record[4:6], "little"),
                            "columns": int.from_bytes(record[6:8], "little"),
                            "level": level,
                            "cells": [],
                        }
                    )
                    continue
                if open_tables:
                    current = open_tables[-1]
                    if tag_id == HWP_LIST_HEADER_TAG:
                        position = _hwp_cell_position(record)
                        if position is not None:
                            current["cells"].append({**position, "text": []})
                    elif tag_id == HWP_PARA_TEXT_TAG and current["cells"]:
                        paragraph = _clean_hwp_text(record.decode("utf-16le", errors="ignore"))
                        if paragraph:
                            current["cells"][-1]["text"].append(paragraph)
                    continue
            if tag_id == HWP_PARA_TEXT_TAG:
                paragraph = _clean_hwp_text(record.decode("utf-16le", errors="ignore"))
                if paragraph:
                    paragraphs.append(paragraph)

        while open_tables:
            _flush_hwp_table(open_tables.pop(), options, paragraphs)

    return "\n".join(paragraphs)


# 표 셀의 위치와 병합 칸 수를 읽습니다. 표 셀이 아닌 목록이면 None을 반환합니다.
def _hwp_cell_position(record: bytes) -> dict | None:
    start = HWP_CELL_POSITION_OFFSET
    if len(record) < start + 8:
        return None
    return {
        "column": int.from_bytes(record[start : start + 2], "little"),
        "row": int.from_bytes(record[start + 2 : start + 4], "little"),
        "column_span": int.from_bytes(record[start + 4 : start + 6], "little"),
        "row_span": int.from_bytes(record[start + 6 : start + 8], "little"),
    }


# HWP 본문에서 표를 셀 단위로 읽습니다.
# _read_hwp는 PARA_TEXT만 가져와 표가 한 줄로 뭉개지지만, 파일에는 행·열 수와 셀 좌표가
# 그대로 들어 있어 함께 읽으면 변환이나 인식 없이 격자를 복원할 수 있습니다.
def extract_hwp_tables(path: str | Path) -> list[dict]:
    path = Path(path)
    if not olefile.isOleFile(path):
        raise ValueError("HWP 5.x OLE 문서가 아닙니다.")

    tables: list[dict] = []
    open_tables: list[dict] = []

    with olefile.OleFileIO(path) as hwp:
        if not hwp.exists("FileHeader") or not hwp.exists("BodyText"):
            raise ValueError("HWP FileHeader 또는 BodyText 스트림이 없습니다.")

        file_header = hwp.openstream("FileHeader").read()
        if not file_header.startswith(b"HWP Document File"):
            raise ValueError("지원하지 않는 HWP 문서입니다.")

        compressed = bool(file_header[36] & 1)
        for tag_id, level, record in _iter_hwp_body_records(hwp, compressed):
            # 표보다 얕은 계층이 나오면 그 표는 끝난 것으로 봅니다.
            while open_tables and level < open_tables[-1]["level"]:
                open_tables.pop()

            if tag_id == HWP_TABLE_TAG and len(record) >= 8:
                table = {
                    "rows": int.from_bytes(record[4:6], "little"),
                    "columns": int.from_bytes(record[6:8], "little"),
                    "level": level,
                    "depth": len(open_tables),
                    "cells": [],
                }
                tables.append(table)
                open_tables.append(table)
                continue

            if not open_tables:
                continue

            current = open_tables[-1]
            if tag_id == HWP_LIST_HEADER_TAG:
                position = _hwp_cell_position(record)
                if position is not None:
                    current["cells"].append({**position, "text": []})
            elif tag_id == HWP_PARA_TEXT_TAG and current["cells"]:
                paragraph = _clean_hwp_text(record.decode("utf-16le", errors="ignore"))
                if paragraph:
                    current["cells"][-1]["text"].append(paragraph)

    for table in tables:
        table.pop("level")
    return tables


# 셀에 적힌 좌표를 그대로 써서 격자에 배치합니다.
# 순서대로 채우면 병합된 칸에서 그 뒤가 한 칸씩 밀립니다.
# 병합된 칸은 첫 칸에만 값을 두고 나머지는 비워 원본 모양을 유지합니다.
def hwp_table_to_markdown(table: dict) -> str:
    rows, columns = table["rows"], table["columns"]
    if rows <= 0 or columns <= 0:
        return ""

    grid = [["" for _ in range(columns)] for _ in range(rows)]
    for cell in table["cells"]:
        if cell["row"] >= rows or cell["column"] >= columns:
            continue
        text = " ".join(cell["text"])
        # 좌표가 같은 셀이 드물게 있습니다. 덮어쓰면 앞의 내용이 사라지므로 이어 붙입니다.
        seat = grid[cell["row"]][cell["column"]]
        grid[cell["row"]][cell["column"]] = f"{seat} {text}".strip() if seat else text
    return "\n".join("| " + " | ".join(row) + " |" for row in grid)


# 셀 중 내용이 있는 셀의 비율입니다. 값을 적지 않은 빈 서식을 걸러낼 때 씁니다.
# 분모는 행×열이 아니라 실제 셀 수입니다. 병합된 셀은 격자에서 여러 칸을 차지하지만
# 셀 자체는 하나여서, 행×열로 나누면 내용이 꽉 찬 표도 밀도가 낮게 나옵니다.
def hwp_table_density(table: dict) -> float:
    cells = table["cells"]
    if not cells:
        return 0.0
    return sum(1 for cell in cells if cell["text"]) / len(cells)


# 청크에 넣을 만한 표인지 판단합니다.
# HWP는 제목에 테두리를 넣거나 여백을 맞추려고 1행·1열 표를 자주 쓰고,
# 값을 적지 않은 빈 서식은 검색 근거가 되지 못합니다.
def is_hwp_content_table(table: dict, min_rows: int, min_columns: int, min_density: float) -> bool:
    if table["rows"] < min_rows or table["columns"] < min_columns:
        return False
    return hwp_table_density(table) >= min_density


# PDF의 표·이미지·텍스트 추출 상태를 점검해 추가 추출 검토 후보를 찾습니다.
def inspect_pdf_text_extraction(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file was not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files can be inspected: {path}")

    reader = PdfReader(str(path))
    page_text_lengths = []
    empty_page_numbers = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        page_text_lengths.append(len(page_text))
        if not page_text:
            empty_page_numbers.append(page_number)

    image_xrefs = set()
    image_page_numbers = []
    table_count = 0
    table_page_numbers = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            page_image_xrefs = {image[0] for image in page.get_images(full=True)}
            image_xrefs.update(page_image_xrefs)
            if page_image_xrefs:
                image_page_numbers.append(page_number)

            found_tables = page.find_tables().tables
            table_count += len(found_tables)
            if found_tables:
                table_page_numbers.append(page_number)

    image_count = len(image_xrefs)
    page_count = len(page_text_lengths)
    empty_page_count = len(empty_page_numbers)

    if empty_page_count == page_count and page_count:
        ocr_recommendation = "ocr_required"
    elif empty_page_count:
        ocr_recommendation = "review_required"
    else:
        ocr_recommendation = "not_required"

    if image_count and table_count:
        visual_content_recommendation = "table_and_image_review_required"
    elif image_count:
        visual_content_recommendation = "image_ocr_review_required"
    elif table_count:
        visual_content_recommendation = "table_extraction_review_required"
    else:
        visual_content_recommendation = "not_required"

    return {
        "file_name": path.name,
        "document_type": "pdf",
        "page_count": page_count,
        "total_text_length": sum(page_text_lengths),
        "empty_page_count": empty_page_count,
        "empty_page_numbers": empty_page_numbers,
        "table_count": table_count,
        "table_count_basis": "pdf_table_candidate",
        "table_page_numbers": table_page_numbers,
        "image_count": image_count,
        "image_page_numbers": image_page_numbers,
        "ocr_recommendation": ocr_recommendation,
        "visual_content_recommendation": visual_content_recommendation,
    }


# HWP의 표 레코드와 BinData 이미지 파일을 점검해 추가 추출 검토 후보를 찾습니다.
def inspect_hwp_visual_content(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file was not found: {path}")
    if path.suffix.lower() != ".hwp":
        raise ValueError(f"Only HWP files can be inspected: {path}")
    if not olefile.isOleFile(path):
        raise ValueError("HWP 5.x OLE 문서가 아닙니다.")

    with olefile.OleFileIO(path) as hwp:
        if not hwp.exists("FileHeader") or not hwp.exists("BodyText"):
            raise ValueError("HWP FileHeader 또는 BodyText 스트림이 없습니다.")

        file_header = hwp.openstream("FileHeader").read()
        if not file_header.startswith(b"HWP Document File"):
            raise ValueError("지원하지 않는 HWP 문서입니다.")

        compressed = bool(file_header[36] & 1)
        table_count = sum(
            tag_id == HWP_TABLE_TAG
            for tag_id, _level, _record in _iter_hwp_body_records(hwp, compressed)
        )
        image_stream_paths = [
            "/".join(stream_path)
            for stream_path in hwp.listdir()
            if len(stream_path) == 2
            and stream_path[0] == "BinData"
            and Path(stream_path[1]).suffix.lower() in HWP_IMAGE_SUFFIXES
        ]

    image_count = len(image_stream_paths)
    if image_count and table_count:
        visual_content_recommendation = "table_and_image_review_required"
    elif image_count:
        visual_content_recommendation = "image_ocr_review_required"
    elif table_count:
        visual_content_recommendation = "table_extraction_review_required"
    else:
        visual_content_recommendation = "not_required"

    return {
        "file_name": path.name,
        "document_type": "hwp",
        "page_count": None,
        "total_text_length": None,
        "empty_page_count": None,
        "empty_page_numbers": [],
        "table_count": table_count,
        "table_count_basis": "hwp_table_record",
        "table_page_numbers": [],
        "image_count": image_count,
        "image_page_numbers": [],
        "ocr_recommendation": "not_checked",
        "visual_content_recommendation": visual_content_recommendation,
    }


# 파일 형식에 맞춰 표·이미지와 OCR 검토 정보를 반환합니다.
def inspect_document_visual_content(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return inspect_pdf_text_extraction(path)
    if path.suffix.lower() == ".hwp":
        return inspect_hwp_visual_content(path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")

# 파일 확장자에 맞는 방식으로 TXT, PDF, HWP 문서의 본문을 읽습니다.
def read_document(
    path: str | Path,
    exclude_tables: bool = False,
    table_options: dict | None = None,
) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"문서 파일을 찾을 수 없습니다: {path}")

    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8-sig").strip()

    if suffix == ".pdf":

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()

    if suffix == ".hwp":
        return _read_hwp(path, exclude_tables, table_options).strip()

    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")


# 추출한 본문을 지정한 크기와 중첩 길이에 따라 여러 청크로 나눕니다.
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


# 표를 행 단위로 나눕니다.
# chunk_text는 줄바꿈을 공백으로 합쳐 표의 행 구분이 사라지므로 표에는 쓸 수 없습니다.
# 크기를 넘으면 행 경계에서 자르고, 잘린 청크마다 헤더 행을 다시 넣어 항목과 값의 관계를 유지합니다.
def chunk_table(markdown: str, max_chars: int) -> list[str]:
    rows = [row for row in markdown.splitlines() if row.strip()]
    if not rows:
        return []
    if len(markdown) <= max_chars:
        return [markdown]

    header = rows[0]
    chunks = []
    current = [header]
    current_length = len(header)

    for row in rows[1:]:
        # 헤더만 남은 상태에서는 한 행이 상한을 넘더라도 잘라내지 않고 그대로 담습니다.
        if current_length + len(row) + 1 > max_chars and len(current) > 1:
            chunks.append("\n".join(current))
            current = [header, row]
            current_length = len(header) + len(row) + 1
            continue
        current.append(row)
        current_length += len(row) + 1

    if len(current) > 1:
        chunks.append("\n".join(current))
    return chunks


# 이미지 한 장을 청크 하나로 만듭니다. 상한을 넘을 때만 줄 단위로 나눕니다.
def chunk_image(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current: list[str] = []
    current_length = 0
    for line in text.splitlines():
        if current and current_length + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# 이미 추출·클리닝을 마친 본문을 청킹하고 각 청크에 문서 ID와 메타데이터를 연결합니다.
def build_chunks_from_text(
    text: str,
    doc_id: str,
    metadata: dict,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}_chunk_{idx:04d}",
            doc_id=doc_id,
            text=chunk,
            metadata=metadata,
        )
        for idx, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap), start=1)
    ]


# 원본 파일에서 본문을 읽어 청킹하고 각 청크에 문서 ID와 메타데이터를 연결합니다.
def build_chunks(
    file_path: str | Path,
    doc_id: str,
    metadata: dict | None = None,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    path = Path(file_path)
    base_metadata = {
        "file_name": path.name,
        "source_path": str(path),
        **(metadata or {}),
    }

    return build_chunks_from_text(
        read_document(path),
        doc_id,
        base_metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# 생성한 청크 목록을 한 줄에 한 청크씩 JSONL 파일로 저장합니다.
def save_chunks_jsonl(chunks: list[Chunk], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


# 청크 JSONL 레코드가 공통 입력 계약을 지키는지 검증합니다.
def validate_chunk_contract(chunks: list[dict]) -> None:
    for index, chunk in enumerate(chunks, start=1):
        missing_fields = [field for field in CHUNK_REQUIRED_FIELDS if field not in chunk]
        if missing_fields:
            raise ValueError(f"{index}번째 청크에 필수 필드가 없습니다: {', '.join(missing_fields)}")
        if not isinstance(chunk["metadata"], dict):
            raise ValueError(f"{index}번째 청크의 metadata는 객체여야 합니다.")
        missing_metadata_fields = [
            field
            for field in REQUIRED_METADATA_FIELDS
            if not isinstance(chunk["metadata"].get(field), str)
            or not chunk["metadata"][field].strip()
        ]
        if missing_metadata_fields:
            raise ValueError(
                f"{index}번째 청크 metadata에 필수 필드가 없습니다: {', '.join(missing_metadata_fields)}"
            )
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError(f"{index}번째 청크의 text는 비어 있지 않은 문자열이어야 합니다.")


# 저장된 JSONL 파일을 읽어 청크 딕셔너리 목록으로 반환합니다.
def load_chunks_jsonl(path: str | Path, *, validate: bool = False) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    if validate:
        validate_chunk_contract(chunks)
    return chunks


# 실제 원본 데이터 없이 전체 RAG 흐름을 시험할 수 있는 샘플 청크를 만듭니다.
def demo_chunks() -> list[dict]:
    text = (
        "가상 RFP 샘플 문서입니다. 발주기관은 가상디지털진흥원이고, "
        "사업명은 공공 AI 학습지원 플랫폼 구축 사업입니다. 주요 요구사항은 교육과정 추천, "
        "학습 이력 관리, 관리자 통계 화면 제공입니다. 제출 방식은 나라장터 온라인 제출이며, "
        "제출 마감일과 예산은 실제 원본 문서 메타데이터를 기준으로 확인해야 합니다."
    )
    chunk = Chunk(
        chunk_id="demo_doc_chunk_0001",
        doc_id="demo_doc",
        text=text,
        metadata={
            "title": "공공 AI 학습지원 플랫폼 구축 사업",
            "agency": "가상디지털진흥원",
            "file_name": "sample_rfp.txt",
        },
    )
    return [asdict(chunk)]
