#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/publish_dockerhub_images.sh [--tag TAG] [--tag-only] [service ...]

Retag locally built VBOGS images for Docker Hub and push them.

Options:
  --tag TAG                     Destination tag for all images. Overrides VBOGS_IMAGE_TAG.
  --tag-only                    Retag images without pushing.

Services:
  vbogs-torch
  vbogs-preprocess
  vbogs-jax
  vbogs-vbgs-render
  vbogs-pipeline

Environment:
  DOCKERHUB_NAMESPACE          Docker Hub namespace/org. Default: oakleyth
  VBOGS_IMAGE_TAG              Destination tag when --tag is omitted. Default: latest
  VBOGS_LOCAL_TORCH_IMAGE      Source Torch image. Default: local/vbogs-torch
  VBOGS_LOCAL_PREPROCESS_IMAGE Source preprocessing image. Default: local/vbogs-preprocess
  VBOGS_LOCAL_JAX_IMAGE        Source JAX image. Default: local/vbogs-jax
  VBOGS_LOCAL_VBGS_RENDER_IMAGE Source VBGS render image. Default: local/vbogs-vbgs-render
  VBOGS_LOCAL_PIPELINE_IMAGE   Source pipeline image. Default: local/vbogs-pipeline
  VBOGS_TORCH_PUSH_IMAGE       Destination Torch image.
  VBOGS_PREPROCESS_PUSH_IMAGE  Destination preprocessing image.
  VBOGS_JAX_PUSH_IMAGE         Destination JAX image.
  VBOGS_VBGS_RENDER_PUSH_IMAGE Destination VBGS render image.
  VBOGS_PIPELINE_PUSH_IMAGE    Destination pipeline image.
  VBOGS_DOCKER_PUSH_ATTEMPTS   Push attempts per image. Default: 4
  VBOGS_DOCKER_PUSH_RETRY_DELAY_SECONDS
                                Initial retry delay in seconds. Default: 20

Example:
  bash scripts/build_stack_serial.sh
  bash scripts/publish_dockerhub_images.sh --tag latest
EOF
}

push_images=1
image_tag="${VBOGS_IMAGE_TAG:-latest}"
services=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --tag-only)
      push_images=0
      shift
      ;;
    --tag)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --tag" >&2
        usage >&2
        exit 1
      fi
      image_tag="$2"
      shift 2
      ;;
    --tag=*)
      image_tag="${1#*=}"
      shift
      ;;
    --)
      shift
      services+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      services+=("$1")
      shift
      ;;
  esac
done

dockerhub_namespace="${DOCKERHUB_NAMESPACE:-oakleyth}"

if [ -z "${image_tag}" ]; then
  echo "Image tag cannot be empty." >&2
  usage >&2
  exit 1
fi

torch_source="${VBOGS_LOCAL_TORCH_IMAGE:-local/vbogs-torch}"
preprocess_source="${VBOGS_LOCAL_PREPROCESS_IMAGE:-local/vbogs-preprocess}"
jax_source="${VBOGS_LOCAL_JAX_IMAGE:-local/vbogs-jax}"
vbgs_render_source="${VBOGS_LOCAL_VBGS_RENDER_IMAGE:-local/vbogs-vbgs-render}"
pipeline_source="${VBOGS_LOCAL_PIPELINE_IMAGE:-local/vbogs-pipeline}"

torch_target="${VBOGS_TORCH_PUSH_IMAGE:-${dockerhub_namespace}/vbogs-torch:${image_tag}}"
preprocess_target="${VBOGS_PREPROCESS_PUSH_IMAGE:-${dockerhub_namespace}/vbogs-preprocess:${image_tag}}"
jax_target="${VBOGS_JAX_PUSH_IMAGE:-${dockerhub_namespace}/vbogs-jax:${image_tag}}"
vbgs_render_target="${VBOGS_VBGS_RENDER_PUSH_IMAGE:-${dockerhub_namespace}/vbogs-vbgs-render:${image_tag}}"
pipeline_target="${VBOGS_PIPELINE_PUSH_IMAGE:-${dockerhub_namespace}/vbogs-pipeline:${image_tag}}"

if [ "${#services[@]}" -eq 0 ]; then
  services=(vbogs-preprocess vbogs-torch vbogs-jax vbogs-vbgs-render vbogs-pipeline)
fi

push_attempts="${VBOGS_DOCKER_PUSH_ATTEMPTS:-4}"
push_retry_delay_seconds="${VBOGS_DOCKER_PUSH_RETRY_DELAY_SECONDS:-20}"

if ! [[ "${push_attempts}" =~ ^[0-9]+$ ]] || [ "${push_attempts}" -lt 1 ]; then
  echo "VBOGS_DOCKER_PUSH_ATTEMPTS must be a positive integer." >&2
  exit 1
fi

if ! [[ "${push_retry_delay_seconds}" =~ ^[0-9]+$ ]]; then
  echo "VBOGS_DOCKER_PUSH_RETRY_DELAY_SECONDS must be a non-negative integer." >&2
  exit 1
fi

push_with_retries() {
  local target="$1"
  local attempt=1
  local delay_seconds="${push_retry_delay_seconds}"

  while true; do
    if [ "${push_attempts}" -gt 1 ]; then
      echo "Push attempt ${attempt}/${push_attempts}: ${target}"
    fi

    if docker push "${target}"; then
      return 0
    fi

    if [ "${attempt}" -ge "${push_attempts}" ]; then
      echo "Push failed after ${push_attempts} attempt(s): ${target}" >&2
      return 1
    fi

    echo "Push failed for ${target}; retrying in ${delay_seconds}s." >&2
    sleep "${delay_seconds}"
    attempt=$((attempt + 1))
    delay_seconds=$((delay_seconds * 2))
  done
}

publish_image() {
  local service="$1"
  local source="$2"
  local target="$3"

  if [[ "${target}" == *@sha256:* ]]; then
    echo "Refusing to tag ${service} to digest-pinned reference: ${target}" >&2
    echo "Use a mutable tag such as ${dockerhub_namespace}/${service}:latest instead." >&2
    exit 1
  fi

  if ! docker image inspect "${source}" >/dev/null 2>&1; then
    echo "Local source image not found for ${service}: ${source}" >&2
    echo "Build it first, or set the matching VBOGS_LOCAL_*_IMAGE override." >&2
    exit 1
  fi

  echo "Tagging ${source} -> ${target}"
  docker tag "${source}" "${target}"

  if [ "${push_images}" -eq 1 ]; then
    echo "Pushing ${target}"
    push_with_retries "${target}"
  fi
}

for service in "${services[@]}"; do
  case "${service}" in
    vbogs-torch)
      publish_image "${service}" "${torch_source}" "${torch_target}"
      ;;
    vbogs-preprocess)
      publish_image "${service}" "${preprocess_source}" "${preprocess_target}"
      ;;
    vbogs-jax)
      publish_image "${service}" "${jax_source}" "${jax_target}"
      ;;
    vbogs-vbgs-render)
      publish_image "${service}" "${vbgs_render_source}" "${vbgs_render_target}"
      ;;
    vbogs-pipeline)
      publish_image "${service}" "${pipeline_source}" "${pipeline_target}"
      ;;
    *)
      echo "Unknown service: ${service}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo
if [ "${push_images}" -eq 1 ]; then
  echo "Retag and push complete."
else
  echo "Retag complete. Push skipped because --tag-only was set."
fi
