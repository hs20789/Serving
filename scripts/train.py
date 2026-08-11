"""학습 스크립트: 예제 데이터를 만들고 모델을 학습해 state_dict 를 저장한다.

서빙 코드와 완전히 분리되어 있다. 이 파일은 FastAPI 를 import 하지 않고,
FastAPI 는 이 파일을 import 하지 않는다.
둘을 잇는 유일한 접점은 `models/model.pth` 파일과 app/config.py 의 상수다.

실행:
    uv run python -m scripts.train
    uv run python -m scripts.train --epochs 500 --seed 7
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from app.config import CLASS_NAMES, INPUT_DIM, NUM_CLASSES, get_settings
from app.model import TinyClassifier, save_model

# 클래스마다 하나씩, 4차원 공간에서의 중심점.
# 이 중심 주변에 정규분포 노이즈를 뿌려 합성 데이터를 만든다.
CLASS_CENTROIDS: torch.Tensor = torch.tensor(
    [
        [-2.0, -1.0, -2.0, -1.0],  # cold
        [0.0, 1.0, 0.0, 1.0],  # warm
        [2.0, -1.0, 2.0, -1.0],  # hot
    ]
)
NOISE_STD = 0.6


def make_dataset(
    samples_per_class: int = 200, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """(X, y) 형태의 합성 분류 데이터를 만든다.

    X: (samples_per_class * NUM_CLASSES, INPUT_DIM) float32
    y: (samples_per_class * NUM_CLASSES,)           int64  ← CrossEntropyLoss 요구 타입
    """
    generator = torch.Generator().manual_seed(seed)

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for class_index in range(NUM_CLASSES):
        noise = torch.randn(samples_per_class, INPUT_DIM, generator=generator) * NOISE_STD
        features.append(CLASS_CENTROIDS[class_index] + noise)
        labels.append(torch.full((samples_per_class,), class_index, dtype=torch.long))

    return torch.cat(features), torch.cat(labels)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def train(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> TinyClassifier:
    """전체 배치 경사하강법으로 모델을 학습하고 학습된 모델을 돌려준다."""
    torch.manual_seed(seed)  # 가중치 초기화를 재현 가능하게

    model = TinyClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()  # logits 를 그대로 받는다 (softmax 불필요)

    model.train()  # 학습 모드
    for epoch in range(1, epochs + 1):
        logits = model(x)  # 순전파
        loss = loss_fn(logits, y)

        optimizer.zero_grad()  # 이전 스텝의 gradient 초기화
        loss.backward()  # 역전파로 gradient 계산
        optimizer.step()  # 가중치 갱신

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | acc {accuracy(logits, y):.4f}")

    return model


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="TinyClassifier 학습")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--output", type=Path, default=settings.model_path)
    args = parser.parse_args()

    x, y = make_dataset(args.samples_per_class, args.seed)
    print(f"학습 데이터: X={tuple(x.shape)}, y={tuple(y.shape)}, 클래스={CLASS_NAMES}")

    model = train(x, y, epochs=args.epochs, lr=args.lr, seed=args.seed)

    # 학습이 끝난 뒤 최종 정확도를 추론 모드에서 확인한다.
    model.eval()
    with torch.inference_mode():
        final_accuracy = accuracy(model(x), y)
    print(f"최종 학습 정확도: {final_accuracy:.4f}")

    save_model(model, args.output)
    print(f"state_dict 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
