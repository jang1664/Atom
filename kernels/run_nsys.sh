#!/bin/bash

name=$1
nsys profile --force-overwrite=true -o ${name:2}.result $1