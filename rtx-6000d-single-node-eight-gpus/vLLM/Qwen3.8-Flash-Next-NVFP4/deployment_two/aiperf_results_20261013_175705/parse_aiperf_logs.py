# -*- coding: utf-8 -*-
"""Parse NVIDIA AIPerf logs (conc_N_inputlen_N_osl_N.log) into a summary CSV."""
import os
import re
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FNAME_RE = re.compile(r'^conc_(\d+)_inputlen_(\d+)_osl_(\d+)\.log$')
SECTION_RE = re.compile(r'^NVIDIA AIPerf \| LLM Metrics(?::\s*(\w+))?\s*$')
COLS = ['avg', 'min', 'max', 'p99', 'p90', 'p50', 'std']


def parse_log(path):
    """Return {(section, metric_name): {col: raw_value}} for one log file."""
    metrics = {}
    current_section = None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            stripped = raw.strip()
            m = SECTION_RE.match(stripped)
            if m:
                current_section = 'Metrics: ' + m.group(1) if m.group(1) else 'Metrics'
                continue
            if stripped.startswith('│'):
                cells = [c.strip() for c in stripped.split('│')[1:-1]]
                if not cells or cells[0] in ('Metric', ''):
                    continue  # header row / wrapped empty row
                key = (current_section, cells[0])
                if key not in metrics:  # keep first occurrence
                    metrics[key] = dict(zip(COLS, cells[1:]))
    return metrics


def to_num(s):
    if s is None:
        return None
    s = s.replace(',', '').strip()
    if s in ('', 'N/A'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fmt(v):
    return '' if v is None else f'{v:.2f}'


def main():
    rows = []
    for fn in sorted(os.listdir(BASE_DIR)):
        m = FNAME_RE.match(fn)
        if not m:
            continue
        conc, isl, osl = int(m.group(1)), int(m.group(2)), int(m.group(3))
        metrics = parse_log(os.path.join(BASE_DIR, fn))

        def base(name, col):
            return to_num(metrics.get(('Metrics', name), {}).get(col))

        def active(name, col):
            return to_num(metrics.get(('Metrics: Active', name), {}).get(col))

        ttft_avg = base('Time to First Token (ms)', 'avg')
        ttft_p99 = base('Time to First Token (ms)', 'p99')
        itl_avg = base('Inter Token Latency (ms)', 'avg')
        itl_p99 = base('Inter Token Latency (ms)', 'p99')
        output_pu = base('Output Token Throughput Per User', 'avg')
        prefill_pu = active('Active Prefill Throughput Per User', 'avg')
        err = base('Request Error Rate (%)', 'avg')

        prefill_total = prefill_pu * conc if prefill_pu is not None else None
        output_total = output_pu * conc if output_pu is not None else None
        success = 100.0 - err if err is not None else None

        rows.append((conc, isl, osl, ttft_avg, ttft_p99, itl_avg, itl_p99,
                     prefill_pu, output_pu, prefill_total, output_total, success, fn))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    out_path = os.path.join(BASE_DIR, 'aiperf_metrics_summary.csv')
    header = [
        'concurrency',
        'Input Sequence Length (tokens)',
        'Output Sequence Length (tokens)',
        'avg Time to First Token (ms)',
        'p99 Time to First Token (ms)',
        'avg Inter Token Latency (ms)',
        'p99 Inter Token Latency (ms)',
        'Prefill Throughput Per User (tokens/sec/user)',
        'Output Token Throughput Per User (tokens/sec/user)',
        'Prefill Throughput Per User (tokens/sec)',
        'Output Token Throughput Per User (tokens/sec)',
        'Success Rate',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0], r[1], r[2],
                        fmt(r[3]), fmt(r[4]), fmt(r[5]), fmt(r[6]),
                        fmt(r[7]), fmt(r[8]), fmt(r[9]), fmt(r[10]),
                        fmt(r[11]) + ('%' if r[11] is not None else '')])
    print(f'Parsed {len(rows)} log files -> {out_path}')


if __name__ == '__main__':
    main()
