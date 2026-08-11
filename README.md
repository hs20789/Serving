# Serving — PyTorch 모델을 FastAPI로 서빙하기

PyTorch로 학습한 모델을 FastAPI HTTP API로 서빙하는 과정을 처음부터 끝까지 확인하기 위한 학습용 예제 프로젝트입니다.

```
클라이언트 ─HTTP POST─▶ FastAPI ─▶ Pydantic 검증 ─▶ torch.Tensor ─▶ PyTorch 모델 ─▶ 추론 ─▶ JSON 응답
```

## 빠른 시작

```bash
uv sync                              # 1. 가상환경 + 의존성 설치
uv run python -m scripts.train       # 2. 모델 학습 및 models/model.pth 저장
uv run uvicorn app.main:app --reload # 3. 서버 실행 (http://127.0.0.1:8000)
uv run pytest                        # 4. 테스트
```

문서: <http://127.0.0.1:8000/docs> (Pydantic 스키마에서 자동 생성된 Swagger UI)

## 1. 디렉터리 구조

```
.
├── app/
│   ├── __init__.py
│   ├── config.py        설정 + 모델 구조 상수 (학습/서빙 공유)
│   ├── model.py         nn.Module 정의 + state_dict 저장/로드
│   ├── schemas.py       Pydantic 요청/응답 스키마
│   ├── inference.py     Predictor — 텐서 변환 + 추론 실행
│   └── main.py          FastAPI 앱, lifespan, 엔드포인트
├── scripts/
│   ├── __init__.py
│   └── train.py         예제 데이터 생성 + 학습 루프 + 모델 저장
├── models/
│   └── model.pth        학습 산출물 (git 미추적 — 학습으로 재생성)
├── tests/
│   ├── conftest.py      테스트용 모델 학습 + TestClient 픽스처
│   ├── test_train.py    학습 / 저장·로드 테스트
│   └── test_api.py      /health, /predict 테스트
├── pyproject.toml       uv 의존성 + pytest 설정
└── README.md
```

## 2. 각 파일의 역할

| 파일 | 책임 | 알지 못하는 것 |
| --- | --- | --- |
| `app/config.py` | 입력 차원(4), 클래스 목록(`cold`/`warm`/`hot`), 모델 경로, 디바이스 | torch, FastAPI |
| `app/model.py` | 모델 구조(`TinyClassifier`), `save_model` / `load_model` | 학습 방법, HTTP |
| `app/schemas.py` | JSON 계약과 입력 검증 규칙 | torch |
| `app/inference.py` | `list[float]` ↔ `torch.Tensor` 변환, `eval()` + `inference_mode()` 추론 | HTTP |
| `app/main.py` | lifespan으로 모델 lifecycle 관리, 라우팅, 의존성 주입 | 모델 내부 구조 |
| `scripts/train.py` | 합성 데이터 생성, 학습 루프, `state_dict` 저장 | FastAPI |

핵심 설계는 **학습과 서빙이 서로를 import 하지 않는다**는 점입니다. 둘을 잇는 접점은 `models/model.pth` 파일과 `app/config.py`의 상수뿐입니다.

## 3. 모델이 저장되고 로딩되는 과정

**저장 (`scripts/train.py` → `app/model.py:save_model`)**

1. `make_dataset()` 이 클래스별 중심점 주변에 정규분포 노이즈를 뿌려 `X (600, 4)`, `y (600,)` 을 만듭니다.
2. `train()` 이 `Adam` + `CrossEntropyLoss` 로 순전파 → 손실 → `zero_grad()` → `backward()` → `step()` 을 반복합니다.
3. `torch.save(model.state_dict(), path)` — 모델 객체 전체가 아니라 **가중치 텐서만** 저장합니다. 구조는 코드(`TinyClassifier`)에 있고 파일에는 숫자만 있으므로, 코드를 리팩터링해도 가중치를 그대로 쓸 수 있습니다.

**로딩 (`app/model.py:load_model`)**

1. `TinyClassifier()` 로 랜덤 초기화된 빈 모델을 만듭니다 → 구조는 코드에서 옵니다.
2. `torch.load(path, weights_only=True)` 로 가중치를 읽습니다. `weights_only=True` 는 pickle을 통한 임의 코드 실행을 막습니다.
3. `load_state_dict()` 로 가중치를 채우고, `.to(device)`, `.eval()` 로 추론 준비를 마칩니다.

`app/config.py` 의 `INPUT_DIM`/`HIDDEN_DIM`/`NUM_CLASSES` 가 학습 때와 달라지면 `load_state_dict()` 가 shape 불일치로 실패합니다. 그래서 이 상수들을 한 곳에서 공유합니다.

**언제 로딩되는가 (`app/main.py:lifespan`)**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor.from_path(...)  # 시작 시 1회
    yield                                           # ← 이 동안 요청 처리
    app.state.predictor = None                      # 종료 시 1회
```

요청마다 로드하지 않고 `app.state` 에 담아 모든 요청이 공유합니다 (`tests/test_api.py::test_model_is_loaded_only_once` 가 이를 검증합니다). 모델 파일이 없으면 서버는 **시작 시점에 즉시 실패**합니다 — 모델 없이 뜬 서버가 요청마다 500을 뿌리는 것보다 낫기 때문입니다.

## 4. HTTP 요청이 모델 추론까지 도달하는 과정

```
POST /predict  {"features": [-2.0, -1.0, -2.0, -1.0]}
  │
  ├─▶ app/main.py       라우팅: predict(payload, predictor)
  │
  ├─▶ app/schemas.py    PredictRequest 검증
  │                     길이 ≠ 4 이거나 숫자가 아니면 → 422, 여기서 중단
  │
  ├─▶ app/main.py       Depends(get_predictor) → app.state.predictor
  │                     (새로 로드하지 않고 시작 시 만든 객체를 재사용)
  │
  ├─▶ app/inference.py  Predictor.predict()
  │                       torch.tensor(features).unsqueeze(0)  → shape (1, 4)
  │                       model.eval()
  │                       with torch.inference_mode():
  │                           logits = model(x)                → shape (1, 3)
  │                           probs  = softmax(logits, dim=1)
  │                       argmax → 클래스 인덱스, .tolist() → 파이썬 float
  │
  ├─▶ app/main.py       PredictResponse 조립 (인덱스 → 라벨 매핑)
  │
  └─▶ JSON 응답
      {"predicted_class": 0, "predicted_label": "cold",
       "confidence": 0.99999988, "probabilities": {"cold": 0.99999988, ...}}
```

`unsqueeze(0)` 은 `nn.Linear` 가 항상 배치 차원을 기대하기 때문에 필요합니다. `inference_mode()` 는 autograd 그래프를 아예 만들지 않아 `no_grad()` 보다 빠릅니다.

## 5. 서버 실행

```bash
uv run uvicorn app.main:app --reload              # 개발 (코드 변경 시 자동 재시작)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000   # 일반 실행
```

환경 변수로 설정을 덮어쓸 수 있습니다 (`app/config.py` 의 `SERVING_` 접두사):

```bash
SERVING_MODEL_PATH=/tmp/other.pth uv run uvicorn app.main:app
```

## 6. 모델 학습

```bash
uv run python -m scripts.train                                  # 기본값
uv run python -m scripts.train --epochs 500 --lr 0.01 --seed 7  # 옵션 지정
uv run python -m scripts.train --output models/experiment.pth   # 저장 위치 변경
```

출력 예시:

```
학습 데이터: X=(600, 4), y=(600,), 클래스=('cold', 'warm', 'hot')
epoch    1 | loss 1.1017 | acc 0.3900
epoch  300 | loss 0.0002 | acc 1.0000
최종 학습 정확도: 1.0000
state_dict 저장 완료: .../models/model.pth
```

## 7. curl 호출 예제

```bash
# 헬스 체크
curl -s http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"device":"cpu"}

# 추론 — cold
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [-2.0, -1.0, -2.0, -1.0]}'
# {"predicted_class":0,"predicted_label":"cold","confidence":0.9999998807907104,
#  "probabilities":{"cold":0.9999998807907104,"warm":9.86e-8,"hot":1.59e-11}}

# 추론 — hot
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [2.0, -1.0, 2.0, -1.0]}'

# 검증 실패 — 길이가 4가 아님 → 422
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [1.0, 2.0]}'
```

각 클래스의 중심점(가장 확실하게 분류되는 입력):

| 클래스 | features |
| --- | --- |
| `cold` | `[-2.0, -1.0, -2.0, -1.0]` |
| `warm` | `[0.0, 1.0, 0.0, 1.0]` |
| `hot` | `[2.0, -1.0, 2.0, -1.0]` |

## 8. 테스트

```bash
uv run pytest          # 전체
uv run pytest -v       # 테스트 이름까지 표시
uv run pytest tests/test_api.py   # API 테스트만
```

테스트는 `models/model.pth` 에 의존하지 않습니다. `tests/conftest.py` 가 임시 디렉터리에 모델을 직접 학습해 저장한 뒤 `SERVING_MODEL_PATH` 로 앱에 주입합니다.

| 테스트 | 확인하는 것 |
| --- | --- |
| `test_make_dataset_shapes` | 데이터 shape/dtype |
| `test_training_learns_the_task` | 학습이 실제로 수행됨 (정확도 > 0.9) |
| `test_save_and_load_roundtrip` | 저장 후 로드한 모델이 같은 출력을 냄 |
| `test_load_missing_file_raises` | 모델 파일 부재 시 명확한 에러 |
| `test_health` | `GET /health` |
| `test_predict_*` | `POST /predict` 응답 형식, 클래스별 정확도, 422 검증 |
| `test_model_is_loaded_only_once` | 요청 간 Predictor 인스턴스 재사용 |

## 참고

- GPU가 없는 환경을 가정해 `pyproject.toml` 에서 CPU 전용 torch 휠 인덱스를 사용합니다. GPU를 쓰려면 `[[tool.uv.index]]` 의 `pytorch-cpu` 블록과 `[tool.uv.sources]` 를 지우고 `uv sync` 하세요.
- 이 예제는 단일 샘플 추론만 다룹니다. 배치 추론이 필요하면 `PredictRequest.features` 를 `list[list[float]]` 로 바꾸고 `unsqueeze(0)` 을 제거하면 됩니다.
