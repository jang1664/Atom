#!/bin/bash

name="$1"
sections="ComputeWorkloadAnalysis Occupancy SpeedOfLight WarpStateStats LaunchStats LaunchStats SourceCounters InstructionStats SpeedOfLight_HierarchicalSingleRooflineChart MemoryWorkloadAnalysis"
section_flags=""
for section in $sections;
do
  section_flags+="--section $section "
done
cmd="ncu -f -o ${name:2}.result $section_flags $1"
eval $cmd