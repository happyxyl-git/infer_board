#!/bin/bash
# aiperf_auto_test_advanced.sh

set -e

# ============ 配置 ============
MODEL="Qwen3.8-27B"
TOKENIZER="/data/models/unsloth/Qwen3.8-27B-NVFP4"
ENDPOINT_TYPE="chat"
IMAGE_WIDTH_MEAN=600
IMAGE_HEIGHT_MEAN=600
ISL=500
OSL=500
URL="127.0.0.1:8002"
REQUEST_COUNT=5

CONCURRENCY_LIST=(1 2 4 8 16 32)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./aiperf_results_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/summary.csv"
echo "concurrency,status,duration_sec,log_file" > "${SUMMARY_FILE}"

echo "并发数列表: ${CONCURRENCY_LIST[*]}"
echo "结果目录: ${OUTPUT_DIR}"
echo ""

for CONCURRENCY in "${CONCURRENCY_LIST[@]}"; do
    LOG_FILE="${OUTPUT_DIR}/concurrency_${CONCURRENCY}.log"
    echo ">>> [$(date +'%H:%M:%S')] 测试 concurrency=${CONCURRENCY} ..."

    START_TIME=$(date +%s)
    if aiperf profile \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --endpoint-type "${ENDPOINT_TYPE}" \
        --image-width-mean "${IMAGE_WIDTH_MEAN}" \
        --image-height-mean "${IMAGE_HEIGHT_MEAN}" \
        --isl "${ISL}" \
        --osl "${OSL}" \
        --streaming \
        --url "${URL}" \
        --request-count "${REQUEST_COUNT}" \
        --concurrency "${CONCURRENCY}" 2>&1 | tee "${LOG_FILE}"; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED"
    fi
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "${CONCURRENCY},${STATUS},${DURATION},${LOG_FILE}" >> "${SUMMARY_FILE}"
    echo "    -> ${STATUS} (${DURATION}s)"
    echo ""
done

echo "=========================================="
echo " 测试完成，汇总结果："
echo "=========================================="
column -t -s',' "${SUMMARY_FILE}"
echo ""
echo "详细日志: ${OUTPUT_DIR}"
