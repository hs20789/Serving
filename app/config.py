"""프로젝트 전역 설정과 상수.

학습 코드(scripts/train.py)와 서빙 코드(app/*)가 **같은 계약**을 보도록
입력 차원과 클래스 목록을 여기 한 곳에서만 정의한다.
이 값이 어긋나면 state_dict 로드가 실패하므로, 공유하는 것이 중요하다.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 이 파일 기준 프로젝트 루트 (app/config.py -> app -> 루트)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --- 모델 구조를 결정하는 상수 -------------------------------------------------
INPUT_DIM: int = 4
HIDDEN_DIM: int = 16
CLASS_NAMES: tuple[str, ...] = ("cold", "warm", "hot")
NUM_CLASSES: int = len(CLASS_NAMES)


class Settings(BaseSettings):
    """환경 변수로 덮어쓸 수 있는 런타임 설정.

    예) SERVING_MODEL_PATH=/tmp/other.pth uv run uvicorn app.main:app
    """

    # pydantic v2 는 `model_` 로 시작하는 필드명을 예약어로 보고 경고한다.
    # 여기서는 `model_path` 를 쓰고 싶으므로 보호 네임스페이스를 해제한다.
    model_config = SettingsConfigDict(
        env_prefix="SERVING_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "PyTorch Serving Demo"
    model_path: Path = PROJECT_ROOT / "models" / "model.pth"
    device: str = "cpu"


@lru_cache
def get_settings() -> Settings:
    """설정을 한 번만 만들어 재사용한다(요청마다 .env 를 다시 읽지 않도록)."""
    return Settings()
