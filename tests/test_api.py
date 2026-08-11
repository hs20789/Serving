"""FastAPI 엔드포인트 검증.

client 픽스처를 사용하는 것 자체가 "서버가 정상적으로 시작되는지"
(= lifespan 이 모델을 성공적으로 로드하는지) 테스트다.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import CLASS_NAMES
from scripts.train import CLASS_CENTROIDS


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "device": "cpu"}


def test_predict_returns_wellformed_response(client: TestClient) -> None:
    response = client.post("/predict", json={"features": [-2.0, -1.0, -2.0, -1.0]})

    assert response.status_code == 200
    body = response.json()

    assert body["predicted_label"] in CLASS_NAMES
    assert CLASS_NAMES[body["predicted_class"]] == body["predicted_label"]
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("class_index", range(len(CLASS_NAMES)))
def test_predict_matches_class_centroid(client: TestClient, class_index: int) -> None:
    """각 클래스의 중심점을 보내면 그 클래스로 분류되어야 한다."""
    response = client.post(
        "/predict", json={"features": CLASS_CENTROIDS[class_index].tolist()}
    )

    assert response.status_code == 200
    assert response.json()["predicted_label"] == CLASS_NAMES[class_index]


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [1.0, 2.0]},  # 너무 짧음
        {"features": [1.0, 2.0, 3.0, 4.0, 5.0]},  # 너무 김
        {"features": ["a", "b", "c", "d"]},  # 숫자가 아님
        {},  # features 누락
    ],
)
def test_predict_rejects_invalid_input(client: TestClient, payload: dict) -> None:
    """Pydantic 검증이 잘못된 입력을 모델에 닿기 전에 막는다."""
    assert client.post("/predict", json=payload).status_code == 422


def test_model_is_loaded_only_once(client: TestClient) -> None:
    """요청마다 모델을 다시 로드하지 않는지 확인한다.

    두 요청이 동일한 Predictor 인스턴스를 공유해야 한다.
    """
    predictor_before = client.app.state.predictor  # type: ignore[attr-defined]

    client.post("/predict", json={"features": [0.0, 1.0, 0.0, 1.0]})
    client.post("/predict", json={"features": [2.0, -1.0, 2.0, -1.0]})

    assert client.app.state.predictor is predictor_before  # type: ignore[attr-defined]
