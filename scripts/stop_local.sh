#!/bin/bash

set -e
set -o pipefail

VALAN_SESSION=$1

get_descendants ()
{
  local children=$(ps -o pid= --ppid "$1")

  for pid in $children
  do
    get_descendants "$pid"
  done
  if (( $1 != $$ && $1 != $PPID )); then
    echo "$1 "
  fi
}

if [ ! -n "${VALAN_SESSION}" ]; then
  echo "VALAN tmux session not specified. Please specify one of the following."
  echo "Running sessions:"
  tmux list-sessions -F#{session_name}
  exit 1
fi
echo "Shutting down ${VALAN_SESSION}."
processes=''
for C in `tmux list-panes -t "${VALAN_SESSION}" -s -F "#{pane_pid} #{pane_current_command}" 2> /dev/null | grep -v tmux | awk '{print $1}'`; do
  processes+=$(get_descendants $C)
done
if [[ "$processes" != '' ]]; then
  kill -9 "$processes" 2> /dev/null
  tmux kill-session -t "${VALAN_SESSION}"
fi
