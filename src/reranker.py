import os
from typing import List
from dotenv import load_dotenv
from src.retriever import SearchResult

try:
    import cohere
except ImportError:
    cohere = None

class Reranker:
    def __init__(self, config: dict):
        load_dotenv()

        if "retrieval" in config:
            retrieval_config = config["retrieval"]

        else:
            retrieval_config = config
        
        # 기본값 없이 직접 접근하여 설정 누락 시 명확한 에러 발생
        self.rerank_config = retrieval_config["rerank"]
        self.model = self.rerank_config["model"]
        
        self.model_name = self.rerank_config["model_name"]
        self.final_top_k = self.rerank_config["final_top_k"]
        
        if self.model == "cohere":
            if cohere is None:
                raise ImportError("Cohere 라이브러리가 설치되어 있지 않습니다. 'pip install cohere'를 실행해주세요.")
            
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError(".env 파일에 COHERE_API_KEY가 설정되어 있지 않습니다.")
            
            self.client = cohere.Client(api_key)
        else:
            raise ValueError(f"현재 지원하지 않는 Reranker 모델입니다: {self.model}")

    def rerank(self, query: str, results: List[SearchResult], top_k: int = None) -> List[SearchResult]:
        if not results:
            return []

        limit = top_k if top_k is not None else self.final_top_k
        
        if self.model == "cohere":
            docs = [res.text for res in results]
            
            response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=limit
            )
            
            reranked_results = []
            for r in response.results:
                original_idx = r.index
                original_res = results[original_idx]
                
                reranked_results.append(
                    SearchResult(
                        chunk_id=original_res.chunk_id,
                        doc_id=original_res.doc_id,
                        text=original_res.text,
                        metadata=original_res.metadata,
                        score=float(r.relevance_score)
                    )
                )
            return reranked_results
            
        return results