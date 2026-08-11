# syntax=docker/dockerfile:1.7

# uv 바이너리 버전을 고정해 의존성 설치 도구도 재현 가능하게 유지한다.
FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

# 실행 image는 작은 Python base에서 시작한다.
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# 의존성 메타데이터를 먼저 복사해 앱 코드가 바뀌어도 이 레이어를 재사용한다.
COPY pyproject.toml uv.lock ./
# uv와 다운로드 cache는 build 중에만 mount하므로 최종 image layer에 남지 않는다.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --no-create-home \
        --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser app ./app

# 빌드 전에 `uv run python -m scripts.train`으로 생성해야 한다.
# 파일이 없으면 COPY 단계에서 빌드가 실패해 모델 없는 이미지를 방지한다.
COPY --chown=appuser:appuser models/model.pth ./models/model.pth

USER appuser

EXPOSE 8000

# 별도 curl 설치 없이 Python 표준 라이브러리로 기존 endpoint를 확인한다.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
