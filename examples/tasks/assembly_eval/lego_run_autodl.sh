#!/usr/bin/env bash
# LEGO evaluation episodes on an AutoDL GPU host that has the RoCo/BrickSim stack in ~/lego_isaac
# (installed by setup_lego_isaac.sh; see output/lego-domino/agent-04/lego/isaac/). No robot, no station.
#
#   examples/tasks/assembly_eval/lego_run_autodl.sh HOST OUT_NAME -- --scene single_on_brick --policy roco_naive --seeds 0
#   examples/tasks/assembly_eval/lego_run_autodl.sh HOST OUT_NAME --pull        # copy ~/lego_isaac/runs/OUT_NAME back
#
# The run is detached on the host (log: ~/lego_isaac/runs/OUT_NAME/run.log); one Isaac process per call.
set -euo pipefail
HOST=$1 NAME=$2
shift 2
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SSH=(ssh -T -o BatchMode=yes -o ConnectTimeout=30)
if [[ ${1:-} == --pull ]]; then
  mkdir -p "$REPO/runs/lego_eval"
  rsync -az -e "ssh -T -o BatchMode=yes" "$HOST:lego_isaac/runs/$NAME" "$REPO/runs/lego_eval/"
  exit 0
fi
[[ ${1:-} == -- ]] && shift
"${SSH[@]}" "$HOST" "mkdir -p ~/lego_isaac/rsh/src/roboskillhub"
rsync -az --delete -e "ssh -T -o BatchMode=yes" --include='*/' --include='*.py' --include='*.json' --exclude='*' \
  "$REPO/src/roboskillhub/" "$HOST:lego_isaac/rsh/src/roboskillhub/"
ARGS=$(printf '%q ' "$@")
"${SSH[@]}" "$HOST" "set -e; mkdir -p ~/lego_isaac/runs/$NAME; cd ~/lego_isaac
  OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONPATH=\$HOME/lego_isaac/rsh/src nohup env/.venv/bin/python -m \
    roboskillhub.tasks.assembly_eval.lego_isaac_runner --roco \$HOME/lego_isaac/src/RoCo_BrickAssembly \
    --out runs/$NAME $ARGS > runs/$NAME/run.log 2>&1 < /dev/null &
  echo started \$!"
