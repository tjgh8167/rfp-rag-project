from src.openai_chroma_retriever import OpenAIChromaRetriever
from src.retriever import SimpleRetriever


def create_retriever(chunks: list[dict], retrieval_config: dict, profile: str | None = None):
    selected_profile = profile or retrieval_config["active_profile"]
    profiles = retrieval_config["profiles"]

    if selected_profile == "baseline":
        return SimpleRetriever(chunks)
    if selected_profile == "openai":
        return OpenAIChromaRetriever(config = retrieval_config) # top_k가 openai 외부에 있으므로 retrieval_config를 전달하도록 변경

    raise ValueError(f"지원하지 않는 Retrieval 프로필입니다: {selected_profile}")
