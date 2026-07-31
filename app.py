"""BidMate 데모 화면.

질문을 넣으면 답변과 근거 청크를 함께 보여줍니다.
검색·생성 로직은 api_main.run()을 그대로 쓰고 화면만 씌웁니다.
사이드바에서 바꾼 설정은 임시 파일로 넘기므로 config/default.yaml은 수정되지 않습니다.
"""

import copy
import importlib.util
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

from api_main import load_config, run

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# 청크 세트마다 담긴 내용이 달라 검색 결과도 달라집니다. 어느 것으로 답했는지 드러내야 합니다.
# 설정에 있는 것만 고를 수 있게 합니다. 세트는 작업이 진행되며 늘어납니다.
CHUNK_SET_LABELS = {
    "chunks": "① 본문만",
    "chunks_with_extraction": "② 표·이미지 추가",
    "chunks_with_section": "③ 절 제목 추가",
    "chunks_with_table_headline": "④ 표 머리말 추가",
}
# 비교 결과 가장 성적이 좋았던 세트를 기본으로 둡니다.
DEFAULT_CHUNK_KEY = "chunks_with_section"
CHUNK_SET_NOTES = {
    "chunks": "표가 구분자 없이 뭉개져 있고 이미지 내용은 아예 없습니다. 개선 전 상태입니다.",
    "chunks_with_extraction": "①에 표를 격자 그대로, 이미지를 글자로 바꿔 넣었습니다. 표 문항 정답률이 크게 올랐습니다.",
    "chunks_with_section": "②에 '1 사업개요' 같은 절 제목을 청크마다 붙였습니다. 키워드 적중률이 가장 높아 이것을 씁니다.",
    "chunks_with_table_headline": "③에 표마다 기관명·사업명을 붙였습니다. 문서는 잘 찾지만 문서 안에서 구분이 흐려져 답을 못 하는 경우가 있습니다.",
}
# openai 프로필은 청크 파일이 아니라 벡터 DB에서 검색합니다.
# 세트를 바꾸려면 그 세트로 색인한 DB를 함께 가리켜야 합니다.
# 네 DB 모두 컬렉션 이름은 설정의 collection_name과 같아 경로만 바꿉니다.
CHUNK_SET_DB = {
    "chunks": "/data/processed/vector_db/openai/800_120",
    "chunks_with_extraction": "/data/processed/vector_db/openai/extraction_800_120",
    "chunks_with_section": "/data/processed/vector_db/openai/section_800_120",
    "chunks_with_table_headline": "/data/processed/vector_db/openai/tablehead_800_120",
}
SEARCH_METHODS = ["similarity", "mmr", "hybrid"]
# baseline은 개발 초기 확인용이라 화면에서는 뺍니다. local은 아직 구현되지 않았습니다.
PROFILES = ["openai", "local"]
UNAVAILABLE_PROFILES = {"local": "로컬 임베딩 검색은 아직 구현되지 않았습니다."}
# gpt-5 계열의 추론 강도. 영문 값을 그대로 두면 무엇을 고르는지 알기 어렵습니다.
REASONING_LEVELS = {
    "minimal": "최소",
    "low": "낮음",
    "medium": "보통",
    "high": "높음",
}
SOURCE_LABELS = {"table": ("표", "#8B5000"), "image": ("이미지", "#1F5C8B")}
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.jsonl"
QUESTION_TYPES = {
    "single": "단일 문서",
    "numeric": "숫자 계산",
    "table": "표 기반",
    "multi_document": "다문서 비교",
    "follow_up": "후속 질문",
    "unsupported": "근거 없음",
    "image": "이미지 기반",
}


@st.cache_data(show_spinner=False)
def chunk_summary(path: str) -> dict:
    """청크 세트의 구성을 센다. 어떤 설정으로 만든 청크인지 화면에 드러내기 위해서다."""
    counts = {"본문": 0, "표": 0, "이미지": 0}
    file_path = Path(path)
    if not file_path.is_file():
        return counts
    with file_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            source = json.loads(line)["metadata"].get("chunk_source")
            counts["표" if source == "table" else "이미지" if source == "image" else "본문"] += 1
    return counts


@st.cache_data(show_spinner=False)
def db_count(directory: str, collection: str) -> int | None:
    """고른 세트의 벡터 DB에 몇 건이 들어 있는지 센다.

    청크 파일과 DB가 어긋나면 검색 결과가 조용히 달라지므로 화면에 함께 띄운다.
    DB가 없으면 None을 돌려준다.
    """
    database = Path(directory) / "chroma.sqlite3"
    if not database.is_file():
        return None
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "select count(*) from embeddings e join segments s on e.segment_id = s.id"
            " join collections c on s.collection = c.id where c.name = ?",
            (collection,),
        ).fetchone()
    return row[0] if row else 0


@st.cache_data(show_spinner=False)
def rerank_blocker() -> str:
    """리랭크를 지금 쓸 수 있는지 확인한다. 못 쓰면 그 이유를 돌려준다.

    쓸 수 없는 상태로 켜두면 질문을 넣은 뒤에야 오류가 나므로 미리 막는다.
    """
    if importlib.util.find_spec("cohere") is None:
        return "cohere 패키지가 설치되어 있지 않습니다."
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("COHERE_API_KEY"):
        return ".env에 COHERE_API_KEY가 없습니다."
    return ""


@st.cache_data(show_spinner=False)
def load_questions() -> list[dict]:
    """평가에 쓰는 공통질문 13문항. 시연할 때 바로 고를 수 있게 한다."""
    if not QUESTIONS_PATH.is_file():
        return []
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_config(settings: dict) -> str:
    """사이드바 값으로 설정을 만들어 임시 파일에 쓴다. 원본 설정 파일은 건드리지 않는다."""
    config = copy.deepcopy(load_config(str(CONFIG_PATH)))
    config["paths"]["chunks"] = config["paths"][settings["chunk_key"]]
    # 청크 세트를 바꾸면 그 세트로 색인한 벡터 DB도 함께 바꿔야 결과가 맞습니다.
    config["retrieval"]["profiles"]["openai"]["persist_directory"] = CHUNK_SET_DB[
        settings["chunk_key"]
    ]
    config["retrieval"]["active_profile"] = settings["profile"]
    config["retrieval"]["search_method"] = settings["search_method"]
    config["retrieval"]["top_k"] = settings["top_k"]

    if settings["search_method"] == "mmr":
        config["retrieval"]["profiles"]["openai"]["fetch_k"] = settings["fetch_k"]
        config["retrieval"]["profiles"]["openai"]["lambda_mult"] = settings["lambda_mult"]
    elif settings["search_method"] == "hybrid":
        config["retrieval"]["hybrid"]["alpha"] = settings["alpha"]
        config["retrieval"]["hybrid"]["candidate_count"] = settings["candidate_count"]

    # 리랭크는 위에서 고른 검색 방식을 대체하지 않고 그 결과를 다시 줄 세웁니다.
    config["retrieval"]["use_rerank"] = settings["use_rerank"]
    if settings["use_rerank"]:
        config["retrieval"]["rerank"]["candidate_count"] = settings["rerank_candidate"]
        # 리랭크를 켜면 최종 개수를 rerank의 final_top_k가 결정합니다.
        # 반환 개수 슬라이더와 뜻이 갈리지 않도록 같은 값을 넣습니다.
        config["retrieval"]["rerank"]["final_top_k"] = settings["top_k"]

    config["generation"]["model"] = settings["model"]
    config["generation"]["max_tokens"] = settings["max_tokens"]
    config["generation"]["reasoning_effort"] = settings["reasoning_effort"]
    config["generation"]["history"]["max_turns"] = settings["max_turns"]

    temporary = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(config, temporary, allow_unicode=True)
    temporary.close()
    return temporary.name


def find_common_question(question: str) -> dict | None:
    """공통질문이면 정답 정보를 찾아 채점에 쓴다. 자유 질문이면 채점하지 않는다."""
    target = " ".join(question.split())
    for item in load_questions():
        if " ".join(item["question"].split()) == target:
            return item
    return None


def score_answer(item: dict, answer: str, sources: list[dict]) -> dict:
    """공통질문의 정답 문서와 키워드가 실제로 잡혔는지 센다."""
    return {
        "question_id": item["question_id"],
        "doc_hit": sum(1 for s in sources if s["doc_id"] in item["expected_doc_ids"]),
        "doc_total": len(sources),
        "keyword_hit": sum(1 for word in item["keywords"] if word in answer),
        "keyword_total": len(item["keywords"]),
    }


def answer_summary(message: dict) -> str:
    """답변마다 그때 쓴 설정을 남긴다. 질문 사이에 설정을 바꿔가며 비교하기 때문이다."""
    used = message["settings"]
    parts = [
        used["search_method"],
        f"top_k {used['top_k']}",
        CHUNK_SET_LABELS[used["chunk_key"]],
    ]
    if used["search_method"] == "mmr":
        parts.append(f"fetch_k {used['fetch_k']} · λ {used['lambda_mult']}")
    elif used["search_method"] == "hybrid":
        parts.append(f"alpha {used['alpha']} · 후보 {used['candidate_count']}")
    # 리랭크는 순서를 바꾸므로 켰는지 여부를 반드시 남깁니다.
    if used["use_rerank"]:
        parts.append(f"리랭크 후보 {used['rerank_candidate']}")
    # 생성 설정도 답변을 바꾸므로 함께 남깁니다.
    parts.append(f"max_tokens {used['max_tokens']:,}")
    parts.append(f"추론 {REASONING_LEVELS[used['reasoning_effort']]}")
    parts.append(f"{message['elapsed']:.1f}초")
    parts.append(f"${message['cost']:.5f}")
    parts.append(f"{message['prompt_tokens']:,}+{message['completion_tokens']:,} 토큰")

    score = message.get("score")
    if score:
        parts.append(
            f"**{score['question_id']}** 정답문서 {score['doc_hit']}/{score['doc_total']}"
            f" · 키워드 {score['keyword_hit']}/{score['keyword_total']}"
        )
    return " · ".join(parts)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("검색된 근거가 없습니다.")
        return

    with st.expander(f"근거 {len(sources)}건", expanded=False):
        for order, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            label, color = SOURCE_LABELS.get(metadata.get("chunk_source"), ("본문", "#4A4A4A"))
            if metadata.get("chunk_source") == "image" and metadata.get("image_type"):
                label = f"이미지·{metadata['image_type']}"

            score = source.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
            st.markdown(
                f"**{order}.** "
                f"<span style='background:{color}; color:white; padding:1px 6px; "
                f"border-radius:4px; font-size:12px;'>{label}</span> "
                f"{metadata.get('agency', '기관 정보 없음')} · "
                f"`{source.get('chunk_id', '-')}` · score {score_text}",
                unsafe_allow_html=True,
            )
            st.caption(metadata.get("file_name", "출처 없음"))
            # 표는 줄바꿈과 구분자가 살아 있어야 격자로 읽힙니다. 마크다운으로 렌더링하면 무너집니다.
            st.text(source["text"][:1200])
            if order < len(sources):
                st.divider()


st.set_page_config(page_title="BidMate", page_icon="📄", layout="wide")
st.title("BidMate")
st.caption("공공입찰 제안요청서(RFP)에서 근거를 찾아 답합니다.")

with st.sidebar:
    st.header("설정")

    profile = st.selectbox("검색 프로필", PROFILES)
    if profile in UNAVAILABLE_PROFILES:
        st.error(UNAVAILABLE_PROFILES[profile])

    base_config = load_config(str(CONFIG_PATH))
    available = {
        key: label
        for key, label in CHUNK_SET_LABELS.items()
        if key in base_config["paths"]
    }
    keys = list(available)
    chunk_key = st.selectbox(
        "청크 세트",
        keys,
        index=keys.index(DEFAULT_CHUNK_KEY) if DEFAULT_CHUNK_KEY in keys else 0,
        format_func=lambda key: available[key],
    )
    chunk_label = available[chunk_key]
    st.caption(CHUNK_SET_NOTES[chunk_key])

    counts = chunk_summary(base_config["paths"][chunk_key])
    st.caption(
        f"본문 {counts['본문']:,} · 표 {counts['표']:,} · 이미지 {counts['이미지']:,}"
        f" · 합계 {sum(counts.values()):,}"
    )

    indexed = db_count(
        CHUNK_SET_DB[chunk_key], base_config["retrieval"]["profiles"]["openai"]["collection_name"]
    )
    if indexed is None:
        st.error("이 세트로 색인한 벡터 DB가 없습니다. 검색 결과가 비어 나옵니다.")
    elif indexed != sum(counts.values()):
        st.warning(f"벡터 DB에는 {indexed:,}건이 들어 있어 청크 파일과 다릅니다.")
    else:
        st.caption(f"벡터 DB {indexed:,}건 색인됨")

    st.divider()
    search_method = st.selectbox("검색 방식", SEARCH_METHODS)
    top_k = st.slider("반환 개수 (top_k)", 1, 20, base_config["retrieval"]["top_k"])

    openai_profile = base_config["retrieval"]["profiles"]["openai"]
    fetch_k = openai_profile["fetch_k"]
    lambda_mult = openai_profile["lambda_mult"]
    alpha = base_config["retrieval"]["hybrid"]["alpha"]
    candidate_count = base_config["retrieval"]["hybrid"]["candidate_count"]

    # 고른 방식에만 해당하는 값을 보여줍니다. 다 띄우면 무엇이 적용되는지 헷갈립니다.
    if search_method == "mmr":
        fetch_k = st.slider("후보 수 (fetch_k)", 10, 300, fetch_k, step=10)
        lambda_mult = st.slider("다양성 가중치 (lambda_mult)", 0.0, 1.0, lambda_mult, step=0.05)
        st.caption("0에 가까울수록 다양성, 1에 가까울수록 관련성을 중시합니다.")
    elif search_method == "hybrid":
        alpha = st.slider("Dense 비율 (alpha)", 0.0, 1.0, alpha, step=0.05)
        candidate_count = st.slider("후보 개수", 10, 200, candidate_count, step=10)
        st.caption("alpha가 클수록 임베딩 유사도, 작을수록 BM25 정확 일치를 중시합니다.")

    # 리랭크는 위 세 방식 중 하나를 고른 뒤 그 결과를 다시 정렬하는 단계라 별도 스위치로 둡니다.
    blocker = rerank_blocker()
    use_rerank = st.checkbox(
        "리랭크 적용 (use_rerank)",
        value=False,
        disabled=bool(blocker),
    )
    rerank_candidate = base_config["retrieval"]["rerank"]["candidate_count"]
    if blocker:
        st.caption(f"지금은 쓸 수 없습니다. {blocker}")
    elif use_rerank:
        rerank_candidate = st.slider("리랭크 후보 개수", 10, 200, rerank_candidate, step=10)
        st.caption(
            f"위 방식으로 {rerank_candidate}건을 먼저 뽑고, "
            f"{base_config['retrieval']['rerank']['model_name']} 모델이 질문과의 관련도로 "
            f"다시 줄을 세워 {top_k}건만 남깁니다."
        )
    else:
        st.caption("켜면 검색 결과를 별도 모델이 질문과의 관련도로 다시 정렬합니다.")

    st.divider()
    st.subheader("답변 생성")
    generation = base_config["generation"]

    # 모델을 바꾸면 벡터 DB·프롬프트와 어긋나 오류가 나므로 표시만 합니다.
    model = generation["model"]
    st.text_input("모델", model, disabled=True)

    max_tokens = st.slider("답변 길이 상한 (max_tokens)", 500, 8000, generation["max_tokens"], step=500)
    st.caption("모델이 한 번에 쓸 수 있는 토큰 수입니다. 내부 추론에도 함께 쓰여 너무 낮으면 답이 비어 나옵니다.")

    reasoning_effort = st.select_slider(
        "추론 강도 (reasoning_effort)",
        options=list(REASONING_LEVELS),
        value=generation["reasoning_effort"],
        format_func=lambda level: REASONING_LEVELS[level],
    )
    st.caption("높일수록 오래 생각하지만 그만큼 답변에 쓸 토큰이 줄어 오히려 빈 답이 나올 수 있습니다.")

    max_turns = st.slider("기억할 대화 턴 수 (history.max_turns)", 0, 10, generation["history"]["max_turns"])
    st.caption("후속 질문에서 앞 대화를 몇 번까지 참고할지 정합니다.")

    st.divider()
    st.subheader("공통질문")
    questions = load_questions()
    if questions:
        chosen = st.selectbox(
            "평가용 13문항",
            questions,
            format_func=lambda item: (
                f"{item['question_id']} · {QUESTION_TYPES.get(item['type'], item['type'])}"
            ),
            label_visibility="collapsed",
        )
        st.caption(chosen["question"])
        if st.button("이 질문 하기", use_container_width=True):
            st.session_state.pending = chosen["question"]
            st.rerun()
    else:
        st.caption("questions.jsonl을 찾지 못했습니다.")

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

settings = {
    "profile": profile,
    "chunk_key": chunk_key,
    "search_method": search_method,
    "top_k": top_k,
    "fetch_k": fetch_k,
    "lambda_mult": lambda_mult,
    "alpha": alpha,
    "candidate_count": candidate_count,
    "use_rerank": use_rerank,
    "rerank_candidate": rerank_candidate,
    "model": model,
    "max_tokens": max_tokens,
    "reasoning_effort": reasoning_effort,
    "max_turns": max_turns,
}

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(answer_summary(message))
            render_sources(message.get("sources", []))

# 공통질문은 사이드바에서 고를 수 있고, 아래 입력창에는 아무 질문이나 넣을 수 있습니다.
question = st.chat_input("제안요청서에 대해 물어보세요") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # rag_engine이 기대하는 형식은 {"question", "answer"} 목록입니다.
    history = [
        {"question": user["content"], "answer": assistant["content"]}
        for user, assistant in zip(
            st.session_state.messages[:-1:2], st.session_state.messages[1::2]
        )
    ]

    with st.chat_message("assistant"):
        with st.spinner("근거를 찾는 중"):
            config_path = build_config(settings)
            start = time.perf_counter()
            # 답변 생성에 쓴 토큰과 비용을 집계합니다. 임베딩 호출은 여기에 잡히지 않습니다.
            with get_openai_callback() as callback:
                response = run(
                    question,
                    config_path=config_path,
                    profile=profile,
                    history=history or None,
                )
            elapsed = time.perf_counter() - start
        st.markdown(response["answer"])
        render_sources(response["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "elapsed": elapsed,
            "cost": callback.total_cost,
            "prompt_tokens": callback.prompt_tokens,
            "completion_tokens": callback.completion_tokens,
            "settings": settings,
            "score": (
                score_answer(common, response["answer"], response["sources"])
                if (common := find_common_question(question))
                else None
            ),
        }
    )
    st.rerun()
