#!/usr/bin/env bash

set -euo pipefail

DEFAULT_DATA_ROOT="/workspace/VBOGS/data/KITTI-360"
if [[ -n "${KITTI_360_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${KITTI_360_DATA_ROOT}"
  DATA_ROOT_FROM_ENV=1
else
  DATA_ROOT="${DEFAULT_DATA_ROOT}"
  DATA_ROOT_FROM_ENV=0
fi
DOWNLOADS_DIR="${KITTI_360_DOWNLOADS_DIR:-"${DATA_ROOT}/_downloads"}"

usage() {
  cat <<'USAGE'
Download and extract KITTI-360 archives into the container stack dataset volume.
Archives are normalized into this layout:
  /workspace/VBOGS/data/KITTI-360/calibration/
  /workspace/VBOGS/data/KITTI-360/data_poses/
  /workspace/VBOGS/data/KITTI-360/images/

Required environment variables:
  KITTI_CALIBRATION_LINK   URL for KITTI-360 calibration archive
  KITTI_POSES_LINK         URL for KITTI-360 poses archive
  KITTI_IMAGES_LINK        URL(s) for KITTI-360 left/right image archive(s),
                           separated by spaces, commas, or newlines

KITTI_CALIBRATION_LINK, KITTI_POSES_LINK, and KITTI_IMAGES_LINK may be omitted
when the matching canonical folders already exist.

Optional:
  KITTI_360_DATA_ROOT       Extraction root. Default:
                            /workspace/VBOGS/data/KITTI-360
                            Expected to be the KITTI-360 Docker volume mount
                            inside vbogs-pipeline.
  KITTI_360_DOWNLOADS_DIR   Archive cache. Default: <KITTI_360_DATA_ROOT>/_downloads
  FORCE_DOWNLOAD=1          Re-download archives even when cached
  KEEP_ARCHIVES=0           Delete cached archives after extraction

Example:
  export KITTI_CALIBRATION_LINK='https://.../calibration.zip'
  export KITTI_POSES_LINK='https://.../data_poses.zip'
  export KITTI_IMAGES_LINK='https://.../data_2d_test_slam.zip'
  ./download_kitti_360.sh
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

get_env_first() {
  local name value
  for name in "$@"; do
    value="$(printenv "${name}" 2>/dev/null || true)"
    if [[ -n "${value}" ]]; then
      printf '%s' "${value}"
      return 0
    fi
  done
  return 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

path_is_mountpoint() {
  local path="$1"

  [[ -r /proc/self/mountinfo ]] || return 2
  awk -v target="${path}" '$5 == target { found = 1 } END { exit found ? 0 : 1 }' /proc/self/mountinfo
}

require_default_data_root_mount() {
  local status

  [[ "${DATA_ROOT_FROM_ENV}" == "0" ]] || return 0

  if [[ ! -d "${DATA_ROOT}" ]]; then
    die "expected KITTI-360 Docker volume mount at ${DATA_ROOT}; run this inside the vbogs-pipeline container or set KITTI_360_DATA_ROOT explicitly"
  fi

  if path_is_mountpoint "${DATA_ROOT}"; then
    return 0
  fi
  status=$?

  if [[ "${status}" == "2" ]]; then
    echo "warning: could not verify whether ${DATA_ROOT} is a Docker volume mount; continuing" >&2
    return 0
  fi

  die "expected ${DATA_ROOT} to be the KITTI-360 Docker volume mount; run this inside the vbogs-pipeline container or set KITTI_360_DATA_ROOT explicitly"
}

archive_name_from_url() {
  local url="$1"
  local no_query base
  no_query="${url%%\?*}"
  no_query="${no_query%%#*}"
  base="${no_query##*/}"
  [[ -n "${base}" ]] || die "could not infer archive name from URL: ${url}"
  printf '%s' "${base}"
}

download_archive() {
  local url="$1"
  local destination="$2"

  if [[ -f "${destination}" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
    echo "Using cached archive: ${destination}"
    return 0
  fi

  mkdir -p "$(dirname "${destination}")"
  echo "Downloading: ${url}"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --show-error --progress-bar --output "${destination}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --output-document="${destination}" "${url}"
  else
    die "either curl or wget is required to download archives"
  fi
}

extract_archive() {
  local archive="$1"
  local destination="$2"

  mkdir -p "${destination}"
  echo "Extracting: ${archive}"

  case "${archive}" in
    *.zip)
      require_command unzip
      unzip -oq "${archive}" -d "${destination}"
      ;;
    *.tar)
      tar -xf "${archive}" -C "${destination}"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${destination}"
      ;;
    *.tar.bz2|*.tbz|*.tbz2)
      tar -xjf "${archive}" -C "${destination}"
      ;;
    *.tar.xz|*.txz)
      tar -xJf "${archive}" -C "${destination}"
      ;;
    *)
      die "unsupported archive format: ${archive}"
      ;;
  esac
}

copy_dir_contents() {
  local source="$1"
  local destination="$2"

  [[ -d "${source}" ]] || return 1
  mkdir -p "${destination}"
  cp -a "${source}/." "${destination}/"
}

copy_pose_drives_from() {
  local source="$1"
  local destination="$2"
  local found=1
  local drive_dir

  [[ -d "${source}" ]] || return 1
  while IFS= read -r -d '' drive_dir; do
    mkdir -p "${destination}/$(basename "${drive_dir}")"
    cp -a "${drive_dir}/." "${destination}/$(basename "${drive_dir}")/"
    found=0
  done < <(find "${source}" -mindepth 1 -maxdepth 1 -type d -name '2013_05_28_drive_*_sync' -print0)

  return "${found}"
}

copy_image_drives_from() {
  local source="$1"
  local destination="$2"
  local found=1
  local drive_dir

  [[ -d "${source}" ]] || return 1
  while IFS= read -r -d '' drive_dir; do
    if [[ -d "${drive_dir}/image_00" || -d "${drive_dir}/image_01" ]]; then
      mkdir -p "${destination}/$(basename "${drive_dir}")"
      cp -a "${drive_dir}/." "${destination}/$(basename "${drive_dir}")/"
      found=0
    fi
  done < <(find "${source}" -mindepth 1 -maxdepth 1 -type d -name '2013_05_28_drive_*_sync' -print0)

  return "${found}"
}

normalize_extraction() {
  local label="$1"
  local extraction_root="$2"
  local target

  case "${label}" in
    calibration)
      target="${DATA_ROOT}/calibration"
      if copy_dir_contents "${extraction_root}/calibration" "${target}"; then
        return 0
      fi
      if [[ -f "${extraction_root}/perspective.txt" ]]; then
        copy_dir_contents "${extraction_root}" "${target}"
        return 0
      fi
      if [[ -d "${extraction_root}/data_3d_semantics/train/2013_05_28_drive_0000_sync" ]]; then
        die "calibration archive did not contain calibration files; check KITTI_CALIBRATION_LINK"
      fi
      copy_dir_contents "${extraction_root}" "${target}"
      ;;
    poses)
      target="${DATA_ROOT}/data_poses"
      if copy_dir_contents "${extraction_root}/data_poses" "${target}"; then
        return 0
      fi
      if copy_pose_drives_from "${extraction_root}" "${target}"; then
        return 0
      fi
      die "poses archive did not contain data_poses or drive pose folders"
      ;;
    images)
      target="${DATA_ROOT}/images"
      if copy_dir_contents "${extraction_root}/images" "${target}"; then
        return 0
      fi
      if copy_image_drives_from "${extraction_root}/data_2d_raw" "${target}"; then
        return 0
      fi
      if copy_image_drives_from "${extraction_root}/data_2d_test" "${target}"; then
        return 0
      fi
      if copy_image_drives_from "${extraction_root}/data_2d_test_slam" "${target}"; then
        return 0
      fi
      if copy_image_drives_from "${extraction_root}" "${target}"; then
        return 0
      fi
      die "images archive did not contain images, data_2d_raw, data_2d_test, data_2d_test_slam, or drive image folders"
      ;;
    *)
      die "unknown archive label: ${label}"
      ;;
  esac
}

split_urls() {
  tr ',\n' '  ' | xargs -n 1 printf '%s\n'
}

has_existing_images() {
  local images_root="$1"
  local found

  [[ -d "${images_root}" ]] || return 1
  found="$(find "${images_root}" -mindepth 2 -maxdepth 2 -type d \( -name 'image_00' -o -name 'image_01' \) -print -quit)"
  [[ -n "${found}" ]]
}

process_url() {
  local label="$1"
  local url="$2"
  local archive_name archive_path extraction_root

  [[ -n "${url}" ]] || die "missing URL for ${label}"
  archive_name="$(archive_name_from_url "${url}")"
  archive_path="${DOWNLOADS_DIR}/${archive_name}"
  extraction_root="$(mktemp -d "${TMPDIR:-/tmp}/kitti360-${label}.XXXXXX")"
  trap 'rm -rf "${extraction_root}"' RETURN

  echo
  echo "==> ${label}"
  download_archive "${url}" "${archive_path}"
  extract_archive "${archive_path}" "${extraction_root}"
  normalize_extraction "${label}" "${extraction_root}"

  if [[ "${KEEP_ARCHIVES:-1}" == "0" ]]; then
    rm -f "${archive_path}"
  fi

  rm -rf "${extraction_root}"
  trap - RETURN
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

calibration_url="$(get_env_first KITTI_CALIBRATION_LINK || true)"
poses_url="$(get_env_first KITTI_POSES_LINK || true)"
images_url="$(get_env_first KITTI_IMAGES_LINK || true)"

missing=()
[[ -n "${calibration_url}" || -d "${DATA_ROOT}/calibration" ]] || missing+=("KITTI_CALIBRATION_LINK")
[[ -n "${poses_url}" || -d "${DATA_ROOT}/data_poses" ]] || missing+=("KITTI_POSES_LINK")
if [[ -z "${images_url}" ]] && ! has_existing_images "${DATA_ROOT}/images"; then
  missing+=("KITTI_IMAGES_LINK")
fi

if (( ${#missing[@]} > 0 )); then
  usage >&2
  echo >&2
  die "missing required environment variable(s): ${missing[*]}"
fi

require_default_data_root_mount
mkdir -p "${DATA_ROOT}" "${DOWNLOADS_DIR}"

echo "KITTI-360 data root : ${DATA_ROOT}"
echo "Archive cache       : ${DOWNLOADS_DIR}"
echo "Canonical layout    : calibration/, data_poses/, images/"

if [[ -n "${calibration_url}" ]]; then
  process_url "calibration" "${calibration_url}"
else
  echo "Using existing calibration: ${DATA_ROOT}/calibration"
fi

if [[ -n "${poses_url}" ]]; then
  process_url "poses" "${poses_url}"
else
  echo "Using existing poses: ${DATA_ROOT}/data_poses"
fi

if [[ -n "${images_url}" ]]; then
  while IFS= read -r image_url; do
    [[ -n "${image_url}" ]] || continue
    process_url "images" "${image_url}"
  done < <(printf '%s\n' "${images_url}" | split_urls)
else
  echo "Using existing images: ${DATA_ROOT}/images"
fi

echo
echo "KITTI-360 downloads extracted into ${DATA_ROOT}"
