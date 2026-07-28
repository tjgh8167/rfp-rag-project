# src/rag_engine.py
import os
from dataclasses import asdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from src.retriever import SearchResult

load_dotenv()

SYSTEM_RULE = (
    "당신은 공공입찰 및 제안요청서(RFP) 분석을 돕는 최고 수준의 전문 AI 어시스턴트입니다.\n"
    "반드시 아래에 제공된 맥락(Context) 문서 내용에만 기반하여 질문에 답변하십시오.\n\n"
    "[작성 원칙]\n"
    "1. 제공된 맥락 내에서만 사실에 기반하여 간결하고 명확하게 답변할 것.\n"
    "2. 주어진 맥락으로 답변이 어려운 경우, 절대 추측하거나 지어내지 말고 '제공된 문서에서 근거를 찾을 수 없어 확인할 수 없습니다'라고 단호하게 답할 것.\n"
    "   맥락 문서가 질문과 같은 산업/분야(예: 교육, 시스템 구축)에 속한다는 이유만으로 관련 있다고 판단하지 말 것. 질문에서 언급한 사업명, 기관명, 핵심 주제어가 맥락 문서에 실제로 등장하지 않으면 근거 없음으로 처리할 것.\n"
    "3. 답변 하단에는 반드시 참고한 문서의 출처(예: 파일명 또는 사업명)를 명시할 것.\n"
    "   답변 본문에서 근거로 사용한 문장 뒤에는 해당 내용을 가져온 문서 번호를 [1], [2]처럼 대괄호로 표기할 것. 여러 문서를 종합했다면 [1][2]처럼 이어서 표기할 것.\n"
    "   한 문장 안에서 인용 번호는 한 번만 표기할 것. 같은 문장의 중간과 끝에 동일한 번호를 중복해서 넣지 말 것.\n"
    "4. 사용자가 읽기 편하도록 적절히 글머리 기호(-, *)를 사용하여 한국어로 작성할 것.\n\n"
    "[대화 이력 처리 원칙]\n"
    "5. 이전 대화(history)가 있다면, 후속 질문에서 기관명·사업명을 다시 언급하지 않아도 직전 대화에서 다룬 기관/사업/주제를 이어받아 답변할 것.\n"
    "   단, 새 질문이 다른 기관명이나 사업명을 명확히 새로 언급하면 이전 대화 내용을 끌어오지 말고 새 질문이 가리키는 대상만을 기준으로 답변할 것.\n\n"
    "[출력 어조 규칙]\n"
    "- 모든 답변은 신뢰감을 주는 **정중한 해요체/하십시오체('~합니다', '~안내해 드립니다')** 또는 **깔끔한 명사형/개조식(~함, ~기재)** 중 하나로 일관되게 작성할 것.\n"
    "- 반말, 혼잣말, 혹은 문장이 중간에 끊기는 현상이 절대 없도록 문장 끝맺음을 완벽하게 마무리할 것."
)


def build_context(results: list[SearchResult]) -> str:
    context_blocks = []
    for idx, result in enumerate(results, start=1):
        metadata = result.metadata
        title = metadata.get("title", "제목 없음")
        agency = metadata.get("agency", "기관 없음")
        source = metadata.get("file_name", "출처 없음")
        context_blocks.append(
            f"[{idx}] title={title} agency={agency} source={source} score={result.score}\n{result.text}"
        )
    return "\n\n".join(context_blocks)


def build_history_messages(history: list[dict] | None, max_turns: int) -> list:
    """이전 대화(history)를 LangChain 메시지로 변환한다. 최근 max_turns턴만 유지해 토큰 증가를 제한한다."""
    if not history:
        return []
    trimmed = history[-max_turns:] if max_turns > 0 else []
    messages = []
    for turn in trimmed:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))
    return messages


def build_llm(config: dict):
    """설정에 맞는 ChatOpenAI 인스턴스를 생성한다. generate_answer와 condense_question이 공유해서 쓴다."""
    gen_config = config["generation"]
    model_name = gen_config["model"]
    temperature = gen_config["temperature"]
    top_p = gen_config["top_p"]
    max_tokens = gen_config["max_tokens"]

    # gpt-5/o-시리즈(reasoning 모델)는 temperature/top_p가 1로 고정되어 있어
    # 커스텀 값을 보내면 400 Unsupported parameter 에러가 발생한다.
    # 해당 모델일 땐 두 파라미터를 빼고, 그 외 모델(gpt-4o 등)일 땐 YAML 값을 그대로 적용한다.
    is_reasoning_model = model_name.startswith(("gpt-5", "o1", "o3", "o4"))
    llm_kwargs = {"model": model_name, "max_tokens": max_tokens}
    if is_reasoning_model:
        llm_kwargs["reasoning_effort"] = gen_config["reasoning_effort"]  # 추가
    else:
        llm_kwargs["temperature"] = temperature
        llm_kwargs["top_p"] = top_p

    return ChatOpenAI(**llm_kwargs)


def condense_question(question: str, history: list[dict] | None, config: dict) -> str:
    """후속 질문을 이전 대화 맥락(기관/사업명 등)을 반영한 독립 질문으로 재구성한다.
    검색(retrieval) 단계에서 이 질문을 사용해야 후속 질문도 올바른 문서를 찾는다."""
    if not history:
        return question
    
    gen_config = config["generation"]
    max_turns = gen_config["history"]["max_turns"]
    recent_history = history[-max_turns:] if max_turns > 0 else []

    if not recent_history:
        return question

    llm = build_llm(config)
    history_text = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent_history
    )
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "아래는 이전 대화 내역입니다. 마지막 질문이 이전 대화의 기관명·사업명 등 문맥을 생략한 "
         "후속 질문이라면, 그 문맥을 반영해 검색에 바로 쓸 수 있는 완전한 독립 질문 하나로 다시 쓰세요. "
         "마지막 질문이 이미 다른 기관·사업을 명확히 새로 언급하고 있다면 원래 질문을 그대로 반환하세요. "
         "재작성된 질문 한 줄만 출력하고 다른 설명은 하지 마세요."),
        ("human", "이전 대화:\n{history}\n\n마지막 질문: {question}\n\n독립 질문:")
    ])
    chain = condense_prompt | llm | StrOutputParser()
    return chain.invoke({"history": history_text, "question": question}).strip()


def generate_answer(
    question: str,
    results: list[SearchResult],
    config: dict,
    history: list[dict] = None,
) -> dict:
    if not results:
        return {
            "answer": "관련 문서 내용을 찾지 못했습니다. 원본 문서나 검색 조건을 다시 확인해 주세요.",
            "sources": [],
        }

    gen_config = config["generation"]
    max_turns = gen_config["history"]["max_turns"]

    llm = build_llm(config)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RULE + "\n\nContext:\n{context}"),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    context = build_context(results)
    history_messages = build_history_messages(history, max_turns)
    answer = chain.invoke({
        "context": context,
        "question": question,
        "history": history_messages,
    })

    return {
        "answer": answer,
        "sources": [asdict(result) for result in results],
        "system_rule": SYSTEM_RULE,
    }