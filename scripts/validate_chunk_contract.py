from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser_chunker import load_chunks_jsonl, validate_chunk_contract
from src.retriever import SearchResult, SimpleRetriever


CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
SAMPLE_CHUNKS_PATH = PROJECT_ROOT / "samples" / "processed" / "sample_chunks.jsonl"
EXPECTED_RESULT_FIELDS = {"chunk_id", "doc_id", "text", "metadata", "score"}


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    chunks = load_chunks_jsonl(SAMPLE_CHUNKS_PATH, validate=True)
    results = SimpleRetriever(chunks).search(
        "제안서 제출",
        top_k=config["retrieval"]["top_k"],
    )

    if not results:
        raise AssertionError("샘플 청크에서 검색 결과를 만들지 못했습니다.")

    result_fields = {field.name for field in fields(SearchResult)}
    if result_fields != EXPECTED_RESULT_FIELDS:
        raise AssertionError(f"SearchResult 필드 불일치: {sorted(result_fields)}")

    invalid_chunks = deepcopy(chunks)
    invalid_chunks[0]["metadata"].pop("agency")
    try:
        validate_chunk_contract(invalid_chunks)
    except ValueError as error:
        if "agency" not in str(error):
            raise
        print(f"누락 필드 검증 성공: {error}")
    else:
        raise AssertionError("필수 metadata 필드 누락을 감지하지 못했습니다.")

    print(f"청크 계약 검증 성공: {len(chunks)}개 청크")
    print(f"SearchResult 필드: {', '.join(sorted(result_fields))}")


if __name__ == "__main__":
    main()
