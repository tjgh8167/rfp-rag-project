import argparse
import json
from pathlib import Path

import yaml

from src.parser_chunker import demo_chunks, load_chunks_jsonl
from src.rag_engine import generate_answer, condense_question
from src.retriever_factory import create_retriever


def load_config(path: str = "config/default.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(chunks_path: str) -> list[dict]:
    path = Path(chunks_path)
    if path.exists():
        return load_chunks_jsonl(path)
    return demo_chunks()


def run(
    question: str,
    config_path: str = "config/default.yaml",
    profile: str | None = None,
    filters: dict | None = None,  # 필터 로직 추가
    history: list[dict] | None = None,
) -> dict:
    config = load_config(config_path)
    chunks = load_chunks(config["paths"]["chunks"])
    retriever = create_retriever(chunks, config["retrieval"], profile)
    selected_profile = profile or config["retrieval"]["active_profile"]

    search_question = condense_question(question, history, config)

    try:
        results = retriever.search(search_question, top_k=config["retrieval"]["top_k"], filters=filters)
    except NotImplementedError as exc:
        return {
            "answer": f"'{selected_profile}' 프로필은 아직 준비되지 않았습니다.\n사유: {exc}",
            "sources": [],
        }

    return generate_answer(question, results, config, history=history)


# 질문/profile/답변/출처(문서·기관·chunk_id·score)를 CLI에 순서대로 정리해 출력한다.
# 검색 결과가 없으면 출처 건수가 0건으로 명확히 표시된다.
def print_result(question: str, profile: str, response: dict) -> None:
    print(f"[질문] {question}")
    print(f"[Profile] {profile}\n")
    print("[답변]")
    print(response["answer"])

    sources = response["sources"]
    print(f"\n[출처 {len(sources)}건]")
    for idx, src in enumerate(sources, start=1):
        metadata = src.get("metadata", {})
        file_name = metadata.get("file_name", "출처 없음")
        agency = metadata.get("agency", "기관 정보 없음")
        chunk_id = src.get("chunk_id", "-")
        score = src.get("score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        print(f"{idx}. {file_name} | 기관: {agency} | chunk_id: {chunk_id} | score: {score_text}")


def main(default_profile: str = "baseline") -> None:
    parser = argparse.ArgumentParser(description="BidMate 최소 End-to-End RAG 실행")
    parser.add_argument("question", nargs="?", default="AI 학습지원 플랫폼의 주요 기능 요구사항을 알려줘")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--profile",
        choices=["baseline", "openai"],
        default=default_profile,
    )
    # 필터 로직 추가
    parser.add_argument("--filters", help="검색 필터를 JSON 형식으로 입력 (예: '{\"agency\": \"가상디지털진흥원\"}')")
    parser.add_argument("--interactive", action="store_true", help="대화형으로 여러 질문을 이어서 물어봅니다.")
    args = parser.parse_args()

    # JSON 문자열을 딕셔너리로 변환, 필터 로직
    filter_dict = json.loads(args.filters) if args.filters else None

    if args.interactive:
        history: list[dict] = []
        print("대화형 모드입니다. 종료하려면 빈 줄을 입력하세요.")
        while True:
            question = input("\n질문> ").strip()
            if not question:
                break
            response = run(question, args.config, args.profile, filter_dict, history=history)
            print_result(question, args.profile, response)
            history.append({"question": question, "answer": response["answer"]})
    else:
        response = run(args.question, args.config, args.profile, filter_dict)
        print_result(args.question, args.profile, response)


if __name__ == "__main__":
    main()
