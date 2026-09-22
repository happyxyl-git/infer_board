# Qwen3.5-27B (SGLang + DFlash2) 服务崩溃日志分析报告

> 分析对象：`qwen3.8-27b-log.txt`（约 5.8MB，31417 行）
> 日志时间范围：2026-09-21 08:03 ~ 2026-09-22 01:06
> 报告生成日期：2026-09-22

---

## 1. 环境概况

| 项目 | 值 |
|---|---|
| 模型 | `Qwen3_5ForConditionalGeneration`（混合线性注意力/Mamba 架构）+ `DFlash2DraftModel` 投机解码草稿模型 |
| 推理框架 | SGLang，单卡 124GB GPU，TP=1 |
| 量化 | compressed-tensors（NVFP4），KV cache 为 `fp8_e4m3` |
| 关键配置 | `mem_fraction_static=0.8`、`max_running_requests=48`、`chunked_prefill_size=2048`、`max_prefill_tokens=16384`、`schedule_policy=fcfs`、`watchdog_timeout=300` |
| 多模态 | 已启用（16 个加载线程 + 2 个隔离处理线程） |

---

## 2. 崩溃时间线

```
09-21 08:03        服务正常启动，权重加载完成（主模型 21.85GB + 草稿模型 3.76GB）
09-21 08:05~09-22 00:30   正常服务（白天负载较低，夜间负载逐渐升高）
09-22 00:28:56     Decode 吞吐已跌至 10.69 tok/s（running-req: 26, mamba usage: 0.65）
09-22 00:33:40     最后一条正常 Decode batch 日志（吞吐 14.95 tok/s）
09-22 00:39:44     最后一条正常 Prefill 日志
                   ↓ 此后调度器前向线程卡死 ≥ 300 秒，无任何日志
09-22 01:05:16     ⚠️ Scheduler watchdog timeout (watchdog_timeout=300, soft=False)
09-22 01:05:16     py-spy 堆栈转储失败（3 次尝试均失败："Failed to copy Py_Version symbol"）
09-22 01:05:21     SIGQUIT received → 触发崩溃诊断 → 进入进程清理
09-22 01:06:27     所有在途 HTTP 请求返回 500
                   （tokenizer_manager._wait_one_response 超时 → CancelledError）
09-22 01:06:27     kill_process_tree → 进程退出
```

---

## 3. 根因分析

### 3.1 直接原因

**调度器前向线程挂起超过 300 秒，触发 watchdog 强杀。**

### 3.2 深层线索（watchdog 触发时的调度器内存池转储）

```
[full]  total=238811, available=12292, evictable=128138, protected=94592
[mamba] total=120,    available=3,     evictable=45,     protected=23
        leaked_full_pages={180226, 180231, 8, 180232, 221193, 8206, ...}  ← 大量泄漏页
```

1. **Mamba 状态池几乎耗尽**
   混合架构模型的线性注意力状态池总共只有 120 个槽位，崩溃时 `available=3`，且转储明确列出了 **`leaked_full_pages`（泄漏的完整页）**——说明有请求退出后 mamba 状态槽位没有被正确释放。

2. **泄漏来源高度可疑是"客户端中断路径"**
   日志统计：
   - **1151 次** `Received output for rid=... but the state was deleted in TokenizerManager`
   - **152 次** `Abort request ... not found in rid_to_state; likely already finished/removed`

   大量客户端超时后主动断开/中止请求，而中止路径在"混合缓存 + 投机解码"场景下未能归还 mamba 槽位（SGLang 对 Qwen3-Next 类混合模型 + speculative decoding 的 abort 清理是已知薄弱点）。

3. **卡死前的雪崩过程**

   ```
   Decode 吞吐从正常水平跌到 10~15 tok/s
   → decode 长时间无产出（00:33 后再无 decode 日志）
   → prefill 继续堆积（pending-token 高达 ~75 万、queue-req 60+）
   → 客户端大面积超时断开（1151 次）
   → 泄漏的 mamba 槽位无法回收
   → 池子耗尽后调度器无法推进
   → watchdog 击杀进程
   ```

### 3.3 结论

**这是 SGLang 在混合线性注意力模型（Qwen3.5）+ DFlash2 投机解码下，请求 abort/断连路径的 mamba 状态池泄漏 bug。长期运行后池资源耗尽，导致调度器活锁挂起，最终被 watchdog 强杀。**

---

## 4. 处理建议

### 4.1 根本解决

- **升级 SGLang 到最新版本**——混合模型缓存管理 + 投机解码的泄漏修复迭代很快，优先查 release notes 中 mamba / hybrid / abort 相关修复。

### 4.2 临时缓解

- 降低 `--max-running-requests`（48 → 24~32），减少同时占用的 mamba 槽位；
- 服务端/客户端同步设置合理的请求超时，避免客户端大量断连触发 abort 路径；
- 若业务允许，可尝试关闭投机解码或换用 EAGLE 对比验证（确认泄漏是否与 DFlash2 相关）。

### 4.3 监控与可观测性

- **监控告警**：日志里的 `mamba usage` 达到 0.6+ 时就该告警，本次崩溃前长期在 0.57~0.72 徘徊；
- **修复 py-spy**：`Failed to copy Py_Version symbol` 导致挂起时拿不到 Python 堆栈，建议在容器内正确安装 py-spy 并匹配 Python 版本，否则下次复现仍无法定位卡死的具体代码行。

### 4.4 泄漏复现验证

- 用压测工具模拟"发出请求后中途断开"的模式，观察 `mamba usage` 是否只增不减，即可在测试环境稳定复现该泄漏并提交 issue。

---

## 5. 关键日志摘录

**Watchdog 触发（第 28958 行）：**

```
[2026-09-22 01:05:16] Scheduler debug info:
[2026-09-22 01:05:16] Pyspy failed (py-spy dump --native --pid 82). Error: Error: Failed to copy Py_Version symbol
[2026-09-22 01:05:16] All pyspy dump attempts failed for PID 82.
[2026-09-22 01:05:16] Scheduler watchdog timeout (self.watchdog_timeout=300, self.soft=False)
[2026-09-22 01:05:21] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
```

**卡死前最后的调度活动（第 29010 行附近，watchdog 后短暂恢复又崩溃）：**

```
[2026-09-22 01:06:11] Prefill batch, #new-seq: 4, #new-token: 2048, #cached-token: 35330,
    full token usage: 0.43, mamba usage: 0.72, #running-req: 26, #queue-req: 55,
    #pending-token: 669940, cuda graph: True, input throughput (token/s): 2667.10
```

**进程退出（第 31534 行）：**

```
[2026-09-22 01:06:27] INFO: 10.10.207.16:43450 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
[2026-09-22 01:06:27] kill_process_tree called: parent_pid=1, include_parent=False, pid=1
```
