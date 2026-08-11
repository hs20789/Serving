"""PyTorch 모델 정의와 state_dict 저장/로드 헬퍼.

이 파일은 "모델이 어떻게 생겼는가"만 책임진다.
학습 방법(scripts/train.py)이나 서빙 방법(app/inference.py)은 알지 못한다.
"""

from pathlib import Path

import torch
from torch import nn

from app.config import HIDDEN_DIM, INPUT_DIM, NUM_CLASSES


class TinyClassifier(nn.Module):
    """4개 특징 -> 3개 클래스를 분류하는 아주 작은 MLP.

    출력은 softmax 를 거치지 않은 **logits** 이다.
    학습에서 쓰는 nn.CrossEntropyLoss 가 내부에서 log_softmax 를 적용하므로
    모델에는 softmax 를 넣지 않고, 확률이 필요한 추론 시점에만 적용한다.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, input_dim) -> (batch_size, num_classes) logits"""
        return self.net(x)


def save_model(model: TinyClassifier, path: Path) -> None:
    """학습된 가중치만(state_dict) 저장한다.

    모델 객체 전체를 pickle 하지 않고 state_dict 만 저장하는 것이 권장 방식이다.
    코드(클래스 정의)와 가중치가 분리되어 있어야 리팩터링 후에도 로드할 수 있다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, device: torch.device | str = "cpu") -> TinyClassifier:
    """저장된 state_dict 를 읽어 추론 준비가 끝난 모델을 돌려준다.

    1) 빈 모델 객체를 만든다 (구조는 코드에서 온다)
    2) 파일에서 가중치를 읽어 넣는다
    3) eval() 로 추론 모드 전환 (Dropout/BatchNorm 동작이 바뀐다)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}\n먼저 `uv run python -m scripts.train` 으로 학습하세요."
        )

    model = TinyClassifier()
    # weights_only=True: 임의 코드 실행 위험이 있는 pickle 객체 대신 텐서만 읽는다.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
