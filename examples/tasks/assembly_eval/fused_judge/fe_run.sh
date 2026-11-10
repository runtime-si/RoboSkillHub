#!/bin/bash
# Copied to a GPU host with the RoCo/BrickSim Isaac venv (~/lego_isaac); see evaluate.py for the analysis.
# fused-judge data: lego_graph_hub (report mode, every verifier runs) per scene x station fault, seeds as given
cd ~/fused-eval
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ROBOSKILLHUB_ROOT=$PWD PYTHONPATH=src:$HOME/lego_isaac/gapdeps
PY=$HOME/lego_isaac/env/.venv/bin/python
scene=$1; fault=$2; shift 2
name=${scene}__${fault}
mkdir -p runs/$name
F=(); [ "$fault" != none ] && F=(--fault $fault)
timeout 5400 $PY examples/tasks/assembly_eval/lego_graph_hub.py --scene $scene --mode report --through-seat "${F[@]}" --out runs/$name --seeds "$@" > runs/$name/run.log 2>&1
echo "== $name rc=$?" >> runs/done.log
