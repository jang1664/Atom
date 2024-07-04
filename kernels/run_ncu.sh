#!/bin/bash

name="$1"
# sections="WarpStateStats SchedulerStats"
# --section WarpStateStats
ncu ncu --section SchedulerStats -f -o ${name:2}.result $1