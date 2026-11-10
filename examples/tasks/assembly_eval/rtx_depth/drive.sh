#!/bin/bash
cd ~/rtx-eval
for f in none perception_offset perception_yaw force_bias stuck_jaws retreat_deviation lost_target; do ./fe_run.sh $1 $f 3 4 5 & sleep 15; done
wait
echo ALL_DONE >> runs/done.log
