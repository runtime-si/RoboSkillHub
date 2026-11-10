#!/bin/bash
# Rendered (LEGO_RENDER=1) variant of examples/tasks/assembly_eval/fused_judge/fe_run.sh; frames under runs/<case>/frames.
# Rendered (LEGO_RENDER=1) variant of examples/tasks/assembly_eval/fused_judge/fe_run.sh; frames under runs/<case>/frames.
# fused-judge data: lego_graph_hub (report mode, every verifier runs) per scene x station fault, seeds as given
cd ~/rtx-eval
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ROBOSKILLHUB_ROOT=$PWD PYTHONPATH=src:$HOME/lego_isaac/gapdeps
PY=$HOME/lego_isaac/env/.venv/bin/python
scene=$1; fault=$2; shift 2
name=${scene}__${fault}__render
mkdir -p runs/$name
F=(); [ "$fault" != none ] && F=(--fault $fault)
LEGO_RENDER=1 LEGO_FRAME_DIR=runs/$name/frames timeout 5400 $PY examples/tasks/assembly_eval/lego_graph_hub.py --scene $scene --mode report --through-seat "${F[@]}" --out runs/$name --seeds "$@" > runs/$name/run.log 2>&1
echo "== $name rc=$?" >> runs/done.log
