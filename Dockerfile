FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

ARG APP_USER=user
ARG APP_UID=1000
ARG APP_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PUMA_CONFIG=/opt/app/configs/submission.toml \
    PUMA_NUCLEI_CHECKPOINT=/opt/app/models/best_primary.ckpt \
    PUMA_TISSUE_CHECKPOINT=/opt/app/models/best_tissue.ckpt

WORKDIR /opt/app

RUN groupadd --gid "${APP_GID}" "${APP_USER}" \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home "${APP_USER}" \
    && mkdir -p /input /output \
    && chown -R "${APP_USER}:${APP_USER}" /opt/app /input /output

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY THIRD_PARTY_PROMETHEUS_LICENSE ./licenses/prometheus/LICENSE
COPY src/ ./src/
RUN python -m pip install --no-cache-dir .

COPY --chown=${APP_USER}:${APP_USER} configs/ ./configs/
COPY --chown=${APP_USER}:${APP_USER} models/best_primary.ckpt ./models/best_primary.ckpt
COPY --chown=${APP_USER}:${APP_USER} models/best_tissue.ckpt ./models/best_tissue.ckpt

RUN python -c "import prometheus, puma_submission" \
    && python -m compileall -q src

USER ${APP_USER}
ENTRYPOINT ["python", "-m", "puma_submission.run"]
