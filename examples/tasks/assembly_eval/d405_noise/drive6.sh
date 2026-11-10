#!/bin/bash
cd ~/rtx-eval
scene=$1; shift
for f in "$@"; do NOISE=fitted ./fe_run.sh $scene $f 3 4 5 & sleep 8; done
wait
echo ALL_DONE_R6 >> runs/done.log
