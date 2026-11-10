#!/bin/bash
# round 7: rendered wrist depth + fitted D405 noise, depth seat with the stud-mask estimator; scene fault seeds...
cd ~/rtx-eval
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ROBOSKILLHUB_ROOT=$PWD PYTHONPATH=src:$HOME/lego_isaac/gapdeps
PY=$HOME/lego_isaac/env/.venv/bin/python
scene=$1; fault=$2; tag=$3; shift 3
name=${scene}__${fault}__r9_${tag}
mkdir -p runs/$name
F=(); [ "$fault" != none ] && F=(--fault $fault)
LEGO_RENDER=1 LEGO_RENDER_NOISE=fitted LEGO_SEAT_ESTIMATOR=stud_mask timeout 3000 $PY examples/tasks/assembly_eval/lego_graph_hub.py --scene $scene --mode report --through-seat "${F[@]}" --out runs/$name --seeds "$@" > runs/$name/run.log 2>&1
