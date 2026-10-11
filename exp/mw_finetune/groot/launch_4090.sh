#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/launch_common.sh" 4090 "$@"
