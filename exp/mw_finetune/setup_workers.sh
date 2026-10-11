#!/usr/bin/env bash
# Read-only system setup: extract OSMesa privately, never apt-install system packages.
set -euo pipefail
root="${MW_ROOT:-/data/mw_finetune}"
libroot="$root/envs/osmesa"
mkdir -p "$libroot"
cd "$libroot"
if [[ ! -e usr/lib/x86_64-linux-gnu/libOSMesa.so.8 ]]; then
  apt-get download libosmesa6
  for package in libosmesa6*.deb; do dpkg-deb -x "$package" .; done
fi
libdir="$libroot/usr/lib/x86_64-linux-gnu"
if [[ ! -e "$libdir/libOSMesa.so" ]]; then
  ln -s libOSMesa.so.8 "$libdir/libOSMesa.so"
fi
printf 'export LD_LIBRARY_PATH=%q:${LD_LIBRARY_PATH:-}\n' "$libdir"
