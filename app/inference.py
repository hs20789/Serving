"""추론 담당 계층: 파이썬 리스트 <-> torch.Tensor 변환과 모델 실행.

FastAPI(app/main.py)는 HTTP 만, 이 파일은 텐서만 다룬다.
이 경계가 있으면 서버 없이도 추론 로직을 단위 테스트할 수 있다.
"""

from collections.abc import Sequence
from pathlib import Path

import torch

from app.config import CLASS_NAMES
from app.model import TinyClassifier, load_model


class Predictor:
    """로드된 모델을 들고 있으면서 요청마다 추론을 수행하는 객체.

    애플리케이션 시작 시 한 번만 생성되고(lifespan), 이후 모든 요청이 공유한다.
    """

    def __init__(self, model: TinyClassifier, device: torch.device) -> None:
        self.model = model
        self.device = device

    @classmethod
    def from_path(cls, path: Path, device: str = "cpu") -> "Predictor":
        """모델 파일 경로에서 Predictor 를 만든다."""
        torch_device = torch.device(device)
        model = load_model(path, torch_device)
        return cls(model, torch_device)

    def predict(self, features: Sequence[float]) -> tuple[int, list[float]]:
        """특징 벡터 하나를 받아 (예측 클래스 인덱스, 클래스별 확률)을 돌려준다."""
        # (input_dim,) 짜리 1차원 입력에 배치 차원을 붙여 (1, input_dim) 으로 만든다.
        # nn.Linear 는 항상 배치 차원을 기대한다.
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        # eval(): Dropout/BatchNorm 을 추론 동작으로 고정 (load_model 에서 이미 호출했지만
        #         추론 직전에 상태를 보장하는 편이 안전하다)
        # inference_mode(): autograd 그래프를 아예 만들지 않아 no_grad 보다 빠르고 메모리도 적다
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

        # 배치 차원을 제거하고 파이썬 타입으로 되돌린다 (JSON 직렬화를 위해).
        probs_1d = probs.squeeze(0)
        predicted_class = int(torch.argmax(probs_1d).item())
        return predicted_class, probs_1d.tolist()

    @staticmethod
    def label_of(class_index: int) -> str:
        return CLASS_NAMES[class_index]

    @staticmethod
    def as_probability_map(probabilities: Sequence[float]) -> dict[str, float]:
        return dict(zip(CLASS_NAMES, probabilities, strict=True))
