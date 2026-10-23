#!/bin/bash
# For every cs_czhai member: mount source, mode, and du (what we can read) of /home/<user>.
one(){ u=$1; d=/home/$u; ls -d $d >/dev/null 2>&1 || { echo "$u NOHOME"; return; }
  src=$(findmnt -no SOURCE $d 2>/dev/null); mode=$(stat -c %A $d 2>/dev/null)
  case "$src" in *cs-czhai-file-01*) ;; *) echo "$u OTHER_FS $src"; return;; esac
  sz=$(timeout 600 du -sxBG $d 2>/dev/null | tail -1 | cut -f1); den=$(timeout 600 du -sx $d 2>&1 >/dev/null | grep -c "Permission denied")
  echo "$u $mode readable=${sz:-?} denied_dirs=$den"; }
export -f one
getent group cs_czhai | sed "s/.*://" | tr "," "\n" | xargs -P 16 -I{} bash -c 'one {}'
