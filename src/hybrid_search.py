import json
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.retriever import SearchResult, tokenize, _metadata_matches

class BM25SparseRetriever:
    def __init__(self, config: dict):
        self.config = config
        
        # config에 paths가 있으면 그걸 쓰고, 없으면 default.yaml을 직접 읽어서 경로를 가져옵니다.
        if "paths" in self.config:
            chunks_path = self.config["paths"]["chunks"]
        else:
            from pathlib import Path
            import yaml
            PROJECT_ROOT = Path.cwd().resolve()
            if not (PROJECT_ROOT / "api_main.py").is_file():
                PROJECT_ROOT = PROJECT_ROOT.parent
                
            # default.yaml 파일 속 paths.chunks 경로 
            config_path = PROJECT_ROOT / "config" / "default.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            
            chunks_path = str(PROJECT_ROOT / full_config["paths"]["chunks"])
            
        self.chunks = self._load_chunks(chunks_path)
        
        # 초기화 시 전체 문서를 토큰화
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        
        # BM25 객체 생성
        self.bm25 = BM25Okapi(tokenized_corpus)

        # retriever.py와 같은 구조 통일
        if "retrieval" in self.config:
            retrieval_config = self.config["retrieval"]
        else:
            retrieval_config = self.config
            
        self.default_top_k = retrieval_config["top_k"]

    def _load_chunks(self, path: str) -> List[Dict[str, Any]]:
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> List[SearchResult]:
        search_k = top_k if top_k is not None else self.default_top_k
        tokenized_query = tokenize(query)

        # BM25 점수 한 번에 연산
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for idx, score in enumerate(doc_scores):
            # 점수가 0 이하인 상관없는 문서 스킵
            if score <= 0:
                continue
                
            chunk = self.chunks[idx]
            metadata = chunk.get("metadata", {})
            
            # 필터 조건 확인 (src/retriever.py의 기존 필터 함수 재활용)
            if not _metadata_matches(metadata, filters):
                continue
                
            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    doc_id=chunk["doc_id"],
                    text=chunk["text"],
                    metadata=metadata,
                    score=float(score)
                )
            )
            
        # 점수 기준 내림차순 정렬 후 상위 top_k개 반환
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:search_k]

class HybridRetriever:
    def __init__(self, config: dict, dense_retriever):
        self.config = config

        if "retrieval" in self.config:
            retrieval_config = self.config["retrieval"]
        else:
            retrieval_config = self.config
        
        # yaml 설정 로드
        self.default_top_k = retrieval_config["top_k"]
        
        hybrid_config = retrieval_config["hybrid"]
        self.alpha = hybrid_config["alpha"]            # 벡터 가중치 (0.0 ~ 1.0)
        self.candidate_count = hybrid_config["candidate_count"] # 후보 추출 수
        
        # 검색기 연결
        self.dense_retriever = dense_retriever              # 넘겨받은 벡터 DB 재활용
        self.sparse_retriever = BM25SparseRetriever(config) # 키워드 엔진 신규 생성

    # 검색 결과 score 정규화
    def _normalize_scores(self, results: List[SearchResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = {}
        for r in results:
            if max_score == min_score:
                norm = 1.0 if max_score > 0 else 0.0
            else:
                norm = (r.score - min_score) / (max_score - min_score)
            normalized[r.chunk_id] = norm
            
        return normalized

    # 최종 반환
    def search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> List[SearchResult]:
        search_k = top_k if top_k is not None else self.default_top_k
        
        # candidate_count 만큼 예비 후보 추출
        dense_results = self.dense_retriever.search(
            query=query, 
            top_k=self.candidate_count, 
            filters=filters, 
            _is_hybrid_internal=True 
        )

        sparse_results = self.sparse_retriever.search(
            query=query,
            top_k=self.candidate_count,
            filters=filters)
        
        # 검색 score 정규화
        dense_norm = self._normalize_scores(dense_results)
        sparse_norm = self._normalize_scores(sparse_results)
        
        # 문서 병합 및 가중치(alpha) 결합 계산
        all_results = {r.chunk_id: r for r in dense_results + sparse_results}
        combined_results = []
        
        for chunk_id, result in all_results.items():
            d_score = dense_norm.get(chunk_id, 0.0) # 벡터에 없으면 0점
            s_score = sparse_norm.get(chunk_id, 0.0) # 키워드에 없으면 0점
            
            # 최종 하이브리드 점수 = (벡터점수 * alpha) + (키워드점수 * (1 - alpha))
            final_score = (d_score * self.alpha) + (s_score * (1.0 - self.alpha))
            
            # SearchResult 계약 유지
            combined_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    text=result.text,
                    metadata=result.metadata,
                    score=round(final_score, 4)
                )
            )
            
        # 점수별로 정렬 된 top_k 반환
        sorted_results = sorted(combined_results, key=lambda x: x.score, reverse=True)
        return sorted_results[:search_k]