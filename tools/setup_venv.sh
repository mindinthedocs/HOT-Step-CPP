#!/usr/bin/env bash
set -euo pipefail

json_info() {
  local msg="$1"
  printf '{"level":"info","message":"%s"}\n' "$msg"
}

json_error() {
  local msg="$1"
  printf '{"level":"error","message":"%s"}\n' "$msg" >&2
}

json_error "not implemented"
exit 2
