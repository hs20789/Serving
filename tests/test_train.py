"""학습 파이프라인과 모델 저장/로드 검증."""

from pathlib import Path

import torch

from app.config import INPUT_DIM, NUM_CLASSES
from app.model import load_model, save_model
from scripts.train import accuracy, make_dataset, train


def test_make_dataset_shapes() -> None:
    x, y = make_dataset(samples_per_class=10, seed=1)

    assert x.shape == (10 * NUM_CLASSES, INPUT_DIM)
    assert y.shape == (10 * NUM_CLASSES,)
    assert y.dtype == torch.long  # CrossEntropyLoss 는 int64 타깃을 요구한다
    assert sorted(y.unique().tolist()) == list(range(NUM_CLASSES))


def test_training_learns_the_task() -> None:
    """학습이 실제로 수행되는지: 학습 후 정확도가 무작위 추측보다 훨씬 높아야 한다."""
    x, y = make_dataset(samples_per_class=100, seed=1)
    model = train(x, y, epochs=200, seed=1, verbose=False)

    model.eval()
    with torch.inference_mode():
        acc = accuracy(model(x), y)

    assert acc > 0.9, f"정확도가 너무 낮습니다: {acc}"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """저장했다 다시 로드한 모델이 동일한 출력을 내는지 확인한다."""
    x, y = make_dataset(samples_per_class=20, seed=2)
    model = train(x, y, epochs=30, seed=2, verbose=False)

    path = tmp_path / "roundtrip.pth"
    save_model(model, path)
    reloaded = load_model(path)

    assert not reloaded.training  # load_model 이 eval() 로 바꿔 놓아야 한다

    model.eval()
    with torch.inference_mode():
        assert torch.allclose(model(x), reloaded(x))


def test_load_missing_file_raises(tmp_path: Path) -> None:
    try:
        load_model(tmp_path / "does-not-exist.pth")
    except FileNotFoundError as exc:
        assert "scripts.train" in str(exc)  # 다음에 뭘 해야 하는지 알려주는 메시지
    else:
        raise AssertionError("FileNotFoundError 가 발생해야 합니다")
