#!/usr/bin/env bash
# Build Docker image and push to Docker Hub.
# Usage:  ./scripts/build_push.sh [tag]
#   tag defaults to "latest"; pass a version like "v1.2" for pinned deploys.
#
# Requires:
#   DOCKER_USER  — Docker Hub username (or set via: export DOCKER_USER=yourname)
#   docker login — run once manually before first push

set -euo pipefail

IMAGE_NAME="${DOCKER_USER:?Set DOCKER_USER env var, e.g. export DOCKER_USER=yourname}/node-agent"
TAG="${1:-latest}"
FULL="${IMAGE_NAME}:${TAG}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Building ${FULL}"
docker build --platform linux/amd64 -t "${FULL}" "${ROOT}"

if [ "${TAG}" != "latest" ]; then
  echo "==> Also tagging as ${IMAGE_NAME}:latest"
  docker tag "${FULL}" "${IMAGE_NAME}:latest"
fi

echo "==> Pushing ${FULL}"
docker push "${FULL}"

[ "${TAG}" != "latest" ] && docker push "${IMAGE_NAME}:latest"

echo ""
echo "Done! Image pushed:"
echo "  ${FULL}"
echo ""
echo "Next: redeploy on AgentBase to pull the new image."
echo "  - Vào dashboard AgentBase → chọn service → Redeploy (hoặc Update image)"
echo "  - Hoặc dùng AgentBase CLI/API nếu có"
