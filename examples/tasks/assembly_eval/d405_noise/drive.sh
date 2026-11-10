#!/bin/bash
cd ~/rtx-eval
for s in single_on_plate single_on_brick; do for f in none perception_offset perception_yaw force_bias stuck_jaws retreat_deviation lost_target; do NOISE=$1 ./fe_run.sh $s $f 3 4 5 & sleep 10; done; done
wait
echo ALL_DONE $1 >> runs/done.log
