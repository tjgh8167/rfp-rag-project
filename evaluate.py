import argparse

from api_main import run


EVALUATION_QUESTIONS = [
    "AI 학습지원 플랫폼의 주요 기능 요구사항을 알려줘",
    "사업 예산과 수행 기간은 어떻게 돼?",
    "제안서는 언제까지 어떤 방식으로 제출해야 해?",
    "문서에 블록체인 기능 요구사항이 있어?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval 프로필별 RAG 평가")
    parser.add_argument(
        "--profile",
        choices=["baseline", "openai"],
        default="baseline",
    )
    args = parser.parse_args()

    for idx, question in enumerate(EVALUATION_QUESTIONS, start=1):
        response = run(question, profile=args.profile)
        print(f"\n[{idx}] 질문: {question}")
        print(response["answer"])


if __name__ == "__main__":
    main()
