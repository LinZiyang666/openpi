#!/usr/bin/env bash
# Take the GR00T fleet down on a simulator box (agents first, then leftovers).
set -uo pipefail
for S in $(tmux ls 2>/dev/null | awk -F: '/^fwgag/{print $1}'); do tmux send-keys -t "$S" C-c 2>/dev/null || true; done
sleep 8
pkill -TERM -f "[r]un_size_eval --role agent" 2>/dev/null || true
sleep 5
left=$(pgrep -fc "examples.libero.[w]orker_entry" || true)
if [ "${left:-0}" -gt 0 ]; then pkill -TERM -f "examples.libero.[w]orker_entry" 2>/dev/null || true; sleep 5; pkill -KILL -f "examples.libero.[w]orker_entry" 2>/dev/null || true; fi
for S in $(tmux ls 2>/dev/null | awk -F: '/^fwgag/{print $1}'); do tmux kill-session -t "$S" 2>/dev/null || true; done
echo "workers left: $(pgrep -fc "examples.libero.[w]orker_entry" || echo 0)"
