"""HTTP 경계에서 오가는 데이터의 모양(Pydantic 스키마).

여기 정의된 타입이 곧 요청 검증 규칙이자 자동 생성되는 OpenAPI 문서다.
torch 는 이 파일을 전혀 모른다 — 순수하게 "JSON 계약"만 담당한다.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.config import INPUT_DIM


class PredictRequest(BaseModel):
    """POST /predict 요청 본문."""

    # 길이가 정확히 INPUT_DIM 이 아니면 FastAPI 가 422 를 돌려준다.
    # 덕분에 잘못된 shape 가 모델까지 내려가지 않는다.
    features: Annotated[
        list[float],
        Field(
            min_length=INPUT_DIM,
            max_length=INPUT_DIM,
            description=f"길이 {INPUT_DIM}의 특징 벡터",
        ),
    ]

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"features": [-1.2, -0.9, -1.1, -0.8]}]}
    )


class PredictResponse(BaseModel):
    """POST /predict 응답 본문."""

    predicted_class: Annotated[int, Field(description="확률이 가장 높은 클래스의 인덱스")]
    predicted_label: Annotated[str, Field(description="클래스 인덱스에 대응하는 이름")]
    confidence: Annotated[float, Field(description="예측 클래스의 확률", ge=0.0, le=1.0)]
    probabilities: Annotated[
        dict[str, float], Field(description="클래스 이름 -> 확률 (합 = 1)")
    ]


class HealthResponse(BaseModel):
    """GET /health 응답 본문."""

    status: str
    model_loaded: bool
    device: str

    # `model_loaded` 가 pydantic 예약 네임스페이스(`model_`)와 겹치는 것을 허용.
    model_config = ConfigDict(protected_namespaces=())
