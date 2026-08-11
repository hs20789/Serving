"""FastAPI 애플리케이션: HTTP 요청/응답과 모델 lifecycle 만 담당한다.

핵심은 lifespan 이다. 모델 로딩은 무거운 작업이므로 요청마다 하면 안 되고,
애플리케이션 시작 시 한 번만 수행해서 app.state 에 담아 모든 요청이 공유한다.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from app.config import get_settings
from app.inference import Predictor
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """yield 이전 = 시작 시 1회, yield 이후 = 종료 시 1회.

    구버전의 @app.on_event("startup") 을 대체하는 현재 권장 방식이다.
    """
    settings = get_settings()
    logger.info("모델 로딩 중: %s (device=%s)", settings.model_path, settings.device)

    # 모델이 없으면 여기서 즉시 실패한다(fail fast).
    # 모델 없이 뜬 서버가 요청을 받아 500 을 뿌리는 것보다 낫다.
    app.state.predictor = Predictor.from_path(settings.model_path, settings.device)
    logger.info("모델 로딩 완료")

    yield  # ---- 이 지점에서 서버가 요청을 처리한다 ----

    logger.info("모델 해제")
    app.state.predictor = None


app = FastAPI(
    title=get_settings().app_name,
    description="PyTorch 모델을 FastAPI 로 서빙하는 최소 예제",
    version="0.1.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> Predictor:
    """lifespan 이 만들어 둔 Predictor 를 꺼내주는 의존성.

    엔드포인트가 app.state 에 직접 손대지 않게 하고, 테스트에서 교체하기도 쉽다.
    """
    return request.app.state.predictor


PredictorDep = Annotated[Predictor, Depends(get_predictor)]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(predictor: PredictorDep) -> HealthResponse:
    """서버와 모델이 요청을 받을 준비가 되었는지 확인한다."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        device=str(predictor.device),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest, predictor: PredictorDep) -> PredictResponse:
    """특징 벡터를 받아 클래스를 예측한다.

    payload 는 이미 Pydantic 검증을 통과한 상태다(길이/타입이 보장됨).
    따라서 여기서는 별도 방어 코드 없이 바로 추론에 넘길 수 있다.
    """
    class_index, probabilities = predictor.predict(payload.features)
    return PredictResponse(
        predicted_class=class_index,
        predicted_label=Predictor.label_of(class_index),
        confidence=probabilities[class_index],
        probabilities=Predictor.as_probability_map(probabilities),
    )
