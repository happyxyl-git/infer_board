# Qwen3.6-27B-NVFP4 性能测试报告

> 测试模型：`unsloth/Qwen3.6-27B-NVFP4`（NVFP4 量化，27B）
> 推理框架：vLLM（镜像 `nvcr.io/nvidia/vllm:26.07-py3`）
> 测试工具：`aiperf profile`（脚本 `perf_test.sh`）
> 测试场景：openai-chat + streaming，ISL=1000 / OSL=1000，并发 1 / 5 / 10

---

## 一、性能指标汇总（avg 值）

| concurrency | Input Sequence Length (tokens) | Output Sequence Length (tokens) | Time to First Token (ms) | Prefill Throughput Per User (tokens/sec/user) | Output Token Throughput Per User (tokens/sec/user) | Request Latency (ms) |
|:-----------:|:------------------------------:|:-------------------------------:|:------------------------:|:---------------------------------------------:|:-------------------------------------------------:|:--------------------:|
| 1           | 1000.00                        | 1000.00                         | 805.28                   | 1590.24                                       | 22.55                                             | 45107.32             |
| 5           | 1000.00                        | 1000.04                         | 1252.75                  | 1099.39                                       | 19.93                                             | 51446.65             |
| 10          | 1000.00                        | 999.90                          | 1381.03                  | 1018.55                                       | 17.03                                             | 60153.24             |

### 简要分析

- **Time to First Token（首 Token 延迟）**：随并发数增加而上升，从 805.28 ms（并发 1）增至 1381.03 ms（并发 10），增幅约 71%。
- **Prefill Throughput Per User（单用户预填充吞吐）**：随并发数增加明显下降，从 1590.24 降至 1018.55 tokens/sec/user（-36%），说明预填充资源被并发请求分摊。
- **Output Token Throughput Per User（单用户解码吞吐）**：随并发数增加缓步下降，从 22.55 降至 17.03 tokens/sec/user（-24%）。
- **Request Latency（请求总延迟）**：随并发数增加持续增长，从 45107.32 ms 增至 60153.24 ms（+33%）。

---

## 二、性能折线图

### 2.1 Time to First Token (ms) vs concurrency

```mermaid
---
config:
  xyChart:
    width: 900
    height: 600
  themeVariables:
    xyChart:
      plotColorPalette: "#00008B"
---
xychart-beta
    title "Time to First Token (ms)"
    x-axis "concurrency" [1, 5, 10]
    y-axis "Time to First Token (ms)" 0 --> 1600
    line [805.28, 1252.75, 1381.03]
```

### 2.2 Prefill Throughput Per User (tokens/sec/user) vs concurrency

```mermaid
---
config:
  xyChart:
    width: 900
    height: 600
  themeVariables:
    xyChart:
      plotColorPalette: "#00008B"
---
xychart-beta
    title "Prefill Throughput Per User (tokens/sec/user)"
    x-axis "concurrency" [1, 5, 10]
    y-axis "tokens/sec/user" 0 --> 1800
    line [1590.24, 1099.39, 1018.55]
```

### 2.3 Output Token Throughput Per User (tokens/sec/user) vs concurrency

```mermaid
---
config:
  xyChart:
    width: 900
    height: 600
  themeVariables:
    xyChart:
      plotColorPalette: "#00008B"
---
xychart-beta
    title "Output Token Throughput Per User (tokens/sec/user)"
    x-axis "concurrency" [1, 5, 10]
    y-axis "tokens/sec/user" 0 --> 25
    line [22.55, 19.93, 17.03]
```

### 2.4 Request Latency (ms) vs concurrency

```mermaid
---
config:
  xyChart:
    width: 900
    height: 600
  themeVariables:
    xyChart:
      plotColorPalette: "#00008B"
---
xychart-beta
    title "Request Latency (ms)"
    x-axis "concurrency" [1, 5, 10]
    y-axis "Request Latency (ms)" 40000 --> 65000
    line [45107.32, 51446.65, 60153.24]
```

---

## 三、部署文档（deployment_infer_service.txt 整理）

### 3.1 推理镜像

```text
nvcr.io/nvidia/vllm:26.07-py3
```

### 3.2 模型权重

```text
/data/models/unsloth/Qwen3.6-27B-NVFP4
```

### 3.3 启动命令

```bash
docker run -it --rm \
  --gpus all \
  -p 8000:8000 \
  -v /data/models/unsloth/Qwen3.6-27B-NVFP4:/model \
  nvcr.io/nvidia/vllm:26.07-py3 \
  vllm serve \
    --model /model \
    --served-model-name Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 100000 \
    --gpu-memory-utilization 0.7 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

### 3.4 启动参数说明

| 参数 | 取值 | 说明 |
|------|------|------|
| `--gpus all` | - | 容器使用宿主机全部 GPU |
| `-p 8000:8000` | - | 映射 vLLM 服务端口 8000 |
| `-v .../Qwen3.6-27B-NVFP4:/model` | - | 将宿主机模型权重目录挂载到容器 `/model` |
| `--model` | `/model` | 模型权重路径（容器内） |
| `--served-model-name` | `Qwen3.6-27B` | 对外提供的服务模型名 |
| `--host` | `0.0.0.0` | 监听所有网卡 |
| `--port` | `8000` | 服务端口 |
| `--tensor-parallel-size` | `1` | 张量并行度（单卡推理） |
| `--max-model-len` | `100000` | 最大模型上下文长度 100K |
| `--gpu-memory-utilization` | `0.7` | 显存利用率上限 70% |
| `--speculative-config` | `mtp, num_speculative_tokens=2` | 投机解码：MTP 方法，每步投机 2 个 token |

---

## 四、性能测试脚本（perf_test.sh 整理）

### 4.1 脚本内容

```bash
NIM_MODEL_NAME="Qwen3.6-27B"
MODEL_PATH="/data/models/unsloth/Qwen3.6-27B-NVFP4"
URL="http://127.0.0.1:8000"
ISL=1000
OSL=1000

# Function to run benchmark
run_benchmark() {
    local CONCURRENCY_COUNT=$1
    echo "========================================"
    echo "Starting benchmark with concurrency: $CONCURRENCY_COUNT"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    aiperf profile \
      --model "$NIM_MODEL_NAME" \
      --tokenizer "$MODEL_PATH" \
      --endpoint-type chat \
      --streaming \
      --url "$URL" \
      --num-requests $((CONCURRENCY_COUNT * 5)) \
      --isl "$ISL" \
      --isl-stddev 0 \
      --osl "$OSL" \
      --osl-stddev 0 \
      --ui-type none \
      --concurrency "$CONCURRENCY_COUNT" \
      --extra-inputs "repetition_penalty:1.0" \
      --extra-inputs "temperature:0.0" \
      --extra-inputs ignore_eos:true \
      --num_dataset_entries $((CONCURRENCY_COUNT * 5)) \
      --no-server-metrics \
      --extra-inputs "min_tokens:40" \
      --tokenizer-trust-remote-code

    echo ""
    echo "Benchmark with concurrency $CONCURRENCY_COUNT completed at $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# Run benchmarks for different concurrency levels
echo "Starting AI Performance Benchmark Suite"
echo "========================================"
echo ""

# 定义需要依次执行的并发档位
CONCURRENCY_LIST=(1 5 10)

# 循环逐个执行压测
for conc in "${CONCURRENCY_LIST[@]}"; do
    run_benchmark "${conc}"
done

echo "========================================"
echo "All benchmarks completed!"
echo "========================================"
```

### 4.2 关键参数说明

| 参数 | 取值 | 说明 |
|------|------|------|
| `NIM_MODEL_NAME` | `Qwen3.6-27B` | 被测服务模型名（与部署 `--served-model-name` 一致） |
| `MODEL_PATH` | `/data/models/unsloth/Qwen3.6-27B-NVFP4` | tokenizer 路径（用于精确统计 token 数） |
| `URL` | `http://127.0.0.1:8000` | 被测服务地址 |
| `ISL` / `OSL` | `1000` / `1000` | 输入 / 输出序列长度（token），标准差为 0 |
| `--num-requests` | `并发数 × 5` | 每档并发的总请求数（如并发 1 → 5 条请求） |
| `--num_dataset_entries` | `并发数 × 5` | 数据集条目数与请求数一致 |
| `--endpoint-type chat` | `chat` | 使用 OpenAI Chat 接口 |
| `--streaming` | - | 流式请求 |
| `--concurrency` | `1 / 5 / 10` | 并发档位（`CONCURRENCY_LIST`） |
| `--extra-inputs repetition_penalty:1.0` | `1.0` | 关闭重复惩罚 |
| `--extra-inputs temperature:0.0` | `0.0` | 贪心解码，保证结果可复现 |
| `--extra-inputs ignore_eos:true` | `true` | 忽略 EOS，强制输出满 OSL |
| `--extra-inputs min_tokens:40` | `40` | 最少生成 40 token |
| `--no-server-metrics` | - | 不采集服务端指标 |
| `--tokenizer-trust-remote-code` | - | 信任远程 tokenizer 代码 |
| `--ui-type none` | `none` | 无 UI 模式运行 |

### 4.3 执行流程

1. 依次遍历并发档位 `CONCURRENCY_LIST=(1 5 10)`；
2. 每档并发发送 `并发数 × 5` 条请求（ISL=1000、OSL=1000、流式）；
3. `aiperf` 输出结果保存至 `perf_test_results/Qwen3.6-27B-openai-chat-concurrency<N>/` 目录（含 `profile_export_aiperf.csv` 等文件）；
4. 全部档位执行完成后输出结束信息。