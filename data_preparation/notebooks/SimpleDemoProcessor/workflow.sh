#!/usr/bin/env bash

set -x -e

# Service environment
WORKFLOW=$(dirname $(readlink -f "$0"))
WORKER_DIR="/home/worker"
IN_DIR="${WORKER_DIR}/workDir/inDir"
OUT_DIR="${WORKER_DIR}/workDir/outDir"
WPS_PROPS="${WORKER_DIR}/workDir/WPS-INPUT.properties"
PROC_DIR="${WORKER_DIR}/procDir"
TIMESTAMP=$(date --utc +%Y%m%d_%H%M%SZ)

# Input parameters available as shell variables
source ${WPS_PROPS}

mkdir -p ${OUT_DIR}/result

echo ${s2_input} >> ${OUT_DIR}/result/in.txt
echo ${s3_input} >> ${OUT_DIR}/result/in.txt

sleep 5
