#!/bin/bash
# docker_build_push.sh
# Build e push das imagens Docker para registry.

set -e

REGISTRY="${DOCKER_REGISTRY:-jacabreu}"
VERSION="${VERSION:-0.2.0}"

echo "Buildando imagens Docker..."

# Build server
echo "Buildando mosaicfl-server..."
docker build -f Dockerfile.server -t "${REGISTRY}/mosaicfl-server:${VERSION}" -t "${REGISTRY}/mosaicfl-server:latest" .

# Build client
echo "Buildando mosaicfl-client..."
docker build -f Dockerfile.client -t "${REGISTRY}/mosaicfl-client:${VERSION}" -t "${REGISTRY}/mosaicfl-client:latest" .

echo ""
echo "Build completo!"
echo ""
echo "Imagens criadas:"
docker images | grep mosaicfl

echo ""
echo "Para publicar no Docker Hub:"
echo "  docker push ${REGISTRY}/mosaicfl-server:${VERSION}"
echo "  docker push ${REGISTRY}/mosaicfl-client:${VERSION}"
echo ""
echo "Para usar localmente:"
echo "  docker-compose up"