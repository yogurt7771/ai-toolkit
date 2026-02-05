#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime
import os
import re
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import html

import gradio as gr
import plotly.graph_objects as go


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# tqdm 行："... 1999/6000 ... loss: 1.203e-01"
STEP_LOSS_RE = re.compile(
    r"\s(?P<step>\d+)\s*/\s*(?P<total>\d+)\s.*?loss:\s*(?P<loss>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
CHECKPOINT_RE = re.compile(r"Saving at step\s+(?P<step>\d+)", re.IGNORECASE)
CKPT_FILE_RE = re.compile(r"_(?P<step>\d{4,})\.safetensors\b", re.IGNORECASE)

# logs/<n>_log.txt
NUMBERED_LOG_RE = re.compile(r"(?P<n>\d+)_log\.txt$", re.IGNORECASE)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def list_run_dirs(base_dir: str, keyword: str = "") -> List[Tuple[str, str]]:
    base = Path(base_dir).expanduser()
    if not base.exists() or not base.is_dir():
        return []
    dirs = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        log_path = p / "log.txt"
        if not log_path.exists() or not log_path.is_file():
            continue
        dirs.append(p)
    if keyword:
        k = keyword.lower()
        dirs = [p for p in dirs if k in p.name.lower()]
    dirs.sort(key=lambda p: (p / "log.txt").stat().st_mtime, reverse=True)
    items: List[Tuple[str, str]] = []
    for p in dirs:
        ts = (p / "log.txt").stat().st_mtime
        dt = datetime.datetime.fromtimestamp(ts)
        label = f"{dt.strftime('%Y-%m-%d %H:%M:%S')} - {p.name}"
        items.append((label, p.name))
    return items


def find_log_parts(run_path: Path) -> List[Path]:
    """
    拼接顺序：
      1) output/<name>/logs/<n>_log.txt   (如果存在，按 n 升序)
      2) output/<name>/log.txt           (如果存在，最后追加)
    """
    parts: List[Tuple[int, Path]] = []
    logs_dir = run_path / "logs"
    if logs_dir.exists() and logs_dir.is_dir():
        for p in logs_dir.iterdir():
            if not p.is_file():
                continue
            m = NUMBERED_LOG_RE.search(p.name)
            if m:
                parts.append((int(m.group("n")), p))

    parts.sort(key=lambda x: x[0])
    ordered: List[Path] = [p for _, p in parts]

    main_log = run_path / "log.txt"
    if main_log.exists() and main_log.is_file():
        ordered.append(main_log)

    return ordered


def parse_log_files(paths: List[Path]) -> Tuple[Dict[int, float], List[int]]:
    """
    Returns:
      loss_by_step: {step: loss}  (同 step 重复时保留“后出现”的，符合拼接后的时间顺序)
      checkpoint_steps: 去重保序
    """
    loss_by_step: Dict[int, float] = {}
    checkpoint_steps_raw: List[int] = []

    for log_path in paths:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = strip_ansi(raw).strip()

                m = STEP_LOSS_RE.search(line)
                if m:
                    step = int(m.group("step"))
                    loss = float(m.group("loss"))
                    loss_by_step[step] = loss  # 保留最后一次

                m2 = CHECKPOINT_RE.search(line)
                if m2:
                    checkpoint_steps_raw.append(int(m2.group("step")))

                m3 = CKPT_FILE_RE.search(line)
                if m3:
                    checkpoint_steps_raw.append(int(m3.group("step")))

    # checkpoint 去重保序
    checkpoint_steps: List[int] = []
    seen = set()
    for s in checkpoint_steps_raw:
        if s not in seen:
            checkpoint_steps.append(s)
            seen.add(s)

    return loss_by_step, checkpoint_steps


def find_logged_neighbors(
    steps_sorted: List[int], cp: int
) -> Tuple[Optional[int], Optional[int]]:
    """
    left  = 日志里 < cp 的最近一步
    right = 日志里 >= cp 的最近一步
    right 可能为 None（比如最后一次 save 后就没 step/loss）
    """
    i = bisect_left(steps_sorted, cp)
    left = steps_sorted[i - 1] if i - 1 >= 0 else None
    right = steps_sorted[i] if i < len(steps_sorted) else None
    return left, right


def build_figure(
    loss_by_step: Dict[int, float],
    checkpoint_steps: List[int],
    yscale: str = "log",
    zoom_window: int = 200,
    neighbor_mode: str = "logged",
) -> Tuple[go.Figure, List[List]]:
    steps_sorted = sorted(loss_by_step.keys())
    losses = [loss_by_step[s] for s in steps_sorted]
    if steps_sorted:
        extra_steps = [steps_sorted[0], steps_sorted[-1]]
        seen = set(checkpoint_steps)
        for s in extra_steps:
            if s not in seen:
                checkpoint_steps.append(s)
                seen.add(s)
        checkpoint_steps.sort()

    rows: List[List] = []  # cp, left_step, left_loss, right_step, right_loss, note
    seg_x: List[Optional[int]] = []
    seg_y: List[Optional[float]] = []
    left_line_x: List[Optional[int]] = []
    left_line_y: List[Optional[float]] = []
    right_line_x: List[Optional[int]] = []
    right_line_y: List[Optional[float]] = []

    pts_x: List[int] = []
    pts_y: List[float] = []
    pts_meta: List[Tuple[int, str]] = []  # (cp, side)

    for cp in checkpoint_steps:
        if neighbor_mode == "numeric":
            left = cp - 1 if (cp - 1) in loss_by_step else None
            right = cp if cp in loss_by_step else None
        else:
            left, right = find_logged_neighbors(steps_sorted, cp)

        note = ""
        left_loss = loss_by_step[left] if left is not None else None
        right_loss = loss_by_step[right] if right is not None else None

        if left is None:
            note += "missing_left; "
        if right is None:
            # 你提醒的情况：save 完就没了
            note += "missing_right(likely last save); "
        note = note.strip()

        rows.append([cp, left, left_loss, right, right_loss, note])

        if left is not None:
            pts_x.append(left)
            pts_y.append(loss_by_step[left])
            pts_meta.append((cp, "before"))
            left_line_x.append(left)
            left_line_y.append(loss_by_step[left])
        else:
            left_line_x.append(None)
            left_line_y.append(None)
        if right is not None:
            pts_x.append(right)
            pts_y.append(loss_by_step[right])
            pts_meta.append((cp, "after_or_at"))
            right_line_x.append(right)
            right_line_y.append(loss_by_step[right])
        else:
            right_line_x.append(None)
            right_line_y.append(None)

        if left is not None and right is not None:
            seg_x += [left, right, None]
            seg_y += [loss_by_step[left], loss_by_step[right], None]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=steps_sorted,
            y=losses,
            mode="lines",
            name="loss (all steps)",
            hovertemplate="step %{x}<br>loss %{y:.6g}<extra></extra>",
            visible="legendonly",
        )
    )

    if seg_x:
        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=seg_y,
                mode="lines",
                name=f"checkpoint neighbor segment ({neighbor_mode})",
                hoverinfo="skip",
            )
        )

    if left_line_x:
        fig.add_trace(
            go.Scatter(
                x=left_line_x,
                y=left_line_y,
                mode="lines",
                name="left neighbor line",
                hoverinfo="skip",
            )
        )

    if right_line_x:
        fig.add_trace(
            go.Scatter(
                x=right_line_x,
                y=right_line_y,
                mode="lines",
                name="right neighbor line",
                hoverinfo="skip",
            )
        )

    if pts_x:
        fig.add_trace(
            go.Scatter(
                x=pts_x,
                y=pts_y,
                mode="markers",
                name="neighbor points",
                customdata=pts_meta,
                hovertemplate="checkpoint %{customdata[0]}<br>side %{customdata[1]}<br>step %{x}<br>loss %{y:.6g}<extra></extra>",
            )
        )

    # checkpoint 竖线
    shapes = []
    for cp in checkpoint_steps:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=cp,
                x1=cp,
                y0=0,
                y1=1,
                line=dict(width=1, dash="dot"),
                opacity=0.35,
            )
        )

    # dropdown：缩放到 ckpt 附近
    buttons = [dict(label="All", method="relayout", args=[{"xaxis.autorange": True}])]
    w = int(zoom_window)
    for cp in checkpoint_steps:
        buttons.append(
            dict(
                label=f"ckpt {cp}",
                method="relayout",
                args=[{"xaxis.range": [cp - w, cp + w]}],
            )
        )

    fig.update_layout(
        title=f"Loss Viewer (checkpoints={len(checkpoint_steps)}, neighbor_mode={neighbor_mode})",
        xaxis=dict(title="step", rangeslider=dict(visible=True), zeroline=False),
        yaxis=dict(
            title="loss", type=("log" if yscale == "log" else "linear"), zeroline=False
        ),
        hovermode="x",
        shapes=shapes,
        updatemenus=[
            dict(
                type="dropdown",
                x=1.01,
                y=1.15,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                showactive=True,
            )
        ],
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=60, r=260, t=120, b=60),
    )

    return fig, rows


def _loss_min_max(rows: List[List]) -> Tuple[Optional[float], Optional[float]]:
    vals: List[float] = []
    for r in rows:
        if len(r) >= 5:
            for v in (r[2], r[4]):
                if isinstance(v, (int, float)):
                    vals.append(float(v))
    if not vals:
        return None, None
    return min(vals), max(vals)


def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def _loss_color(value: Optional[float], vmin: Optional[float], vmax: Optional[float]) -> str:
    if value is None:
        return "#334155"  # slate-700
    if vmin is None or vmax is None or vmin == vmax:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))

    # green -> yellow -> red
    g = (34, 197, 94)
    y = (245, 158, 11)
    r = (239, 68, 68)
    if t < 0.5:
        t2 = t * 2.0
        rgb = (_lerp(g[0], y[0], t2), _lerp(g[1], y[1], t2), _lerp(g[2], y[2], t2))
    else:
        t2 = (t - 0.5) * 2.0
        rgb = (_lerp(y[0], r[0], t2), _lerp(y[1], r[1], t2), _lerp(y[2], r[2], t2))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _fmt_num(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.6g}"


def build_loss_table_html(rows: List[List], headers: List[str]) -> str:
    vmin, vmax = _loss_min_max(rows)
    legend = (
        '<div style="display:flex;align-items:center;gap:8px;margin:6px 0 10px 0;">'
        '<span style="font-size:12px;color:#334155;">loss 低 → 高</span>'
        '<span style="display:inline-block;width:140px;height:10px;'
        'background:linear-gradient(90deg,#22c55e 0%,#f59e0b 50%,#ef4444 100%);'
        'border-radius:6px;border:1px solid rgba(15,23,42,0.15);"></span>'
        "</div>"
    )

    head_cells = "".join(
        f'<th style="text-align:left;padding:8px 10px;border-bottom:1px solid rgba(15,23,42,0.12);'
        f'font-size:12px;color:#0f172a;background:#f8fafc;">{html.escape(h)}</th>'
        for h in headers
    )

    body_rows = []
    for r in rows:
        cp, left_step, left_loss, right_step, right_loss, note = (r + [None] * 6)[:6]
        left_color = _loss_color(left_loss, vmin, vmax)
        right_color = _loss_color(right_loss, vmin, vmax)
        row_html = (
            "<tr>"
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(15,23,42,0.08);color:#0f172a;">{_fmt_num(cp)}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(15,23,42,0.08);color:#0f172a;">{_fmt_num(left_step)}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(15,23,42,0.08);'
            f'background:{left_color};color:#0f172a;font-variant-numeric:tabular-nums;">'
            f'{_fmt_num(left_loss)}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(15,23,42,0.08);color:#0f172a;">{_fmt_num(right_step)}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(15,23,42,0.08);'
            f'background:{right_color};color:#0f172a;font-variant-numeric:tabular-nums;">'
            f'{_fmt_num(right_loss)}</td>'
            f'<td style="padding:6px 10px;border-bottom:1px solid rgba(15,23,42,0.08);color:#475569;">'
            f'{html.escape(str(note)) if note else "-"}</td>'
            "</tr>"
        )
        body_rows.append(row_html)

    table = (
        '<div style="background:#ffffff;border:1px solid rgba(15,23,42,0.12);'
        'border-radius:10px;padding:10px 12px;">'
        f"{legend}"
        '<div style="max-height:520px;overflow:auto;">'
        '<table style="width:100%;border-collapse:collapse;background:#ffffff;">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div></div>"
    )
    return table


def resolve_run_path(base_dir: str, run_name: str) -> Path:
    return (Path(base_dir).expanduser() / run_name).resolve()


def ui_refresh_dirs(base_dir: str, keyword: str):
    dirs = list_run_dirs(base_dir, keyword)
    value = dirs[0][1] if dirs else None
    return gr.Dropdown(choices=dirs, value=value), gr.Markdown(
        f"找到目录：**{len(dirs)}** 个"
    )


def ui_plot(
    base_dir: str, run_name: str, yscale: str, zoom_window: int, neighbor_mode: str
):
    if not run_name:
        raise gr.Error("请先选择一个 output 子目录。")

    run_path = resolve_run_path(base_dir, run_name)
    if not run_path.exists():
        raise gr.Error(f"目录不存在：{run_path}")

    log_parts = find_log_parts(run_path)
    if not log_parts:
        raise gr.Error(
            "没找到 log 文件。期望存在以下之一：\n"
            f"- {run_path / 'log.txt'}\n"
            f"- {run_path / 'logs' / '<n>_log.txt'}"
        )

    loss_by_step, checkpoint_steps = parse_log_files(log_parts)
    if not loss_by_step:
        raise gr.Error(
            "解析不到 step/loss。确认 log 里包含 `1999/6000 ... loss: ...` 这种片段。"
        )

    fig, rows = build_figure(
        loss_by_step=loss_by_step,
        checkpoint_steps=checkpoint_steps,
        yscale=yscale,
        zoom_window=int(zoom_window),
        neighbor_mode=neighbor_mode,
    )

    steps_sorted = sorted(loss_by_step.keys())
    used_files_md = "\n".join([f"- `{p}`" for p in log_parts])

    info = (
        f"**Run**: `{run_path.name}`  \n"
        f"**拼接的 log 文件（按顺序）**：  \n{used_files_md}  \n\n"
        f"解析到 step：**{len(steps_sorted)}**（范围 {steps_sorted[0]} ~ {steps_sorted[-1]}）  \n"
        f"解析到 checkpoint：**{len(checkpoint_steps)}**"
    )

    headers = [
        "checkpoint_step",
        "left_step",
        "left_loss",
        "right_step",
        "right_loss",
        "note",
    ]
    return fig, info, build_loss_table_html(rows, headers)


def build_app(default_base: str) -> gr.Blocks:
    with gr.Blocks(title="Loss Curve Viewer") as demo:
        gr.Markdown("## AI Toolkit Loss Curve Viewer")

        with gr.Row():
            base_dir = gr.Textbox(label="output 基目录", value=default_base, scale=2)
            keyword = gr.Textbox(label="目录过滤关键字（包含匹配）", value="", scale=2)
            refresh_btn = gr.Button("刷新目录", scale=1)

        with gr.Row():
            run_dir = gr.Dropdown(
                label="选择一个 run 目录", choices=[], value=None, scale=3
            )

        with gr.Row():
            yscale = gr.Radio(
                label="Y轴", choices=["log", "linear"], value="log", scale=1
            )
            neighbor_mode = gr.Radio(
                label="checkpoint 相邻步策略",
                choices=["logged", "numeric"],
                value="logged",
                info="logged=按日志实际出现的 step 找左右邻居（能处理最后一次 save 后没 step/loss）",
                scale=2,
            )
            zoom_window = gr.Number(
                label="下拉缩放窗口（±步数）", value=200, precision=0, scale=1
            )
            plot_btn = gr.Button("绘图", scale=1)

        status = gr.Markdown("")
        plot = gr.Plot(label="Loss 曲线（可缩放/悬停/rangeslider）")
        info = gr.Markdown("")
        gr.Markdown("### checkpoint 邻居明细（带色卡）")
        table = gr.HTML()

        demo.load(
            fn=ui_refresh_dirs, inputs=[base_dir, keyword], outputs=[run_dir, status]
        )
        refresh_btn.click(
            fn=ui_refresh_dirs, inputs=[base_dir, keyword], outputs=[run_dir, status]
        )

        plot_btn.click(
            fn=ui_plot,
            inputs=[base_dir, run_dir, yscale, zoom_window, neighbor_mode],
            outputs=[plot, info, table],
        )

    return demo


def resolve_auth_password() -> Optional[str]:
    value = os.getenv("AI_TOOLKIT_AUTH", "").strip()
    return value or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./output", help="output 基目录（默认 ./output）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8676, type=int)
    args = ap.parse_args()

    app = build_app(args.base)
    auth_pwd = resolve_auth_password()
    if auth_pwd:
        app.launch(
            server_name=args.host,
            server_port=args.port,
            auth=lambda u, p: u == "user" and p == auth_pwd,
        )
    else:
        app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
