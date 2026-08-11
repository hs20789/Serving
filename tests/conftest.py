"""테스트 공용 픽스처.

테스트는 저장소에 커밋된 모델 파일에 의존하지 않는다.
대신 세션 시작 시 임시 디렉터리에 모델을 직접 학습해서 저장하고,
SERVING_MODEL_PATH 환경 변수로 앱이 그 파일을 보게 만든다.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.model import save_model
from scripts.train import make_dataset, train


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """테스트용으로 짧게 학습한 모델의 state_dict 경로."""
    x, y = make_dataset(samples_per_class=100, seed=0)
    model = train(x, y, epochs=150, seed=0, verbose=False)

    path = tmp_path_factory.mktemp("models") / "model.pth"
    save_model(model, path)
    return path


@pytest.fixture(scope="session")
def client(trained_model_path: Path) -> Iterator[TestClient]:
    """lifespan 이 실행된 상태의 테스트 클라이언트.

    `with TestClient(app)` 형태로 써야 startup/shutdown(lifespan)이 실행된다.
    그냥 TestClient(app) 만 만들면 모델이 로드되지 않는다.
    """
    os.environ["SERVING_MODEL_PATH"] = str(trained_model_path)
    get_settings.cache_clear()  # 캐시된 이전 설정을 버린다

    # 환경 변수를 세팅한 뒤에 import 해야 앱이 올바른 경로를 본다.
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

    os.environ.pop("SERVING_MODEL_PATH", None)
    get_settings.cache_clear()
