"""
Shared helpers for the demo report generators (build_report.py,
build_facesurv_report.py): image thumbnailing, the per-frame similarity
chart, and the page shell/CSS so both reports look and behave the same.
"""
import base64
import os

import cv2
import numpy as np

BLUE = "#2a78d6"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"


def thumb_data_uri(path, max_w=220):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def evenly_spaced(items, n):
    items = list(items)
    if len(items) <= n:
        return items
    idx = sorted(set(np.linspace(0, len(items) - 1, n).round().astype(int)))
    return [items[i] for i in idx]


def img_strip(uris, css_class="thumb"):
    return "".join(f'<img class="{css_class}" src="{u}">' for u in uris if u)


def build_similarity_chart(members, avg_similarity, threshold, avg_label="track avg",
                            width=760, height=220):
    """members: list of dicts with 'own_similarity' and a label key ('timestamp' or 'index')."""
    pad_l, pad_r, pad_t, pad_b = 40, 20, 20, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(members)
    bar_gap = 6
    bar_w = max(4, (plot_w - bar_gap * (n - 1)) / n) if n else 0

    def y_of(v):
        return pad_t + plot_h * (1 - max(0.0, min(1.0, v)))

    bars = []
    for i, m in enumerate(members):
        x = pad_l + i * (bar_w + bar_gap)
        v = m["own_similarity"]
        y = y_of(v)
        h = pad_t + plot_h - y
        label = m.get("chart_label", str(i))
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
            f'rx="2" fill="var(--series-1)"><title>{label}  similarity={v:.3f}</title></rect>'
        )

    def ref_line(v, dash, label, color):
        y = y_of(v)
        return (
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
            f'<text x="{width - pad_r}" y="{y - 4:.1f}" text-anchor="end" '
            f'class="ref-label" fill="{color}">{label} {v:.3f}</text>'
        )

    axis = (
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="var(--baseline)"/>'
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="var(--baseline)"/>'
    )
    ticks = "".join(
        f'<text x="{pad_l - 8}" y="{y_of(v) + 4:.1f}" text-anchor="end" class="axis-label">{v:.1f}</text>'
        f'<line x1="{pad_l}" y1="{y_of(v):.1f}" x2="{width - pad_r}" y2="{y_of(v):.1f}" stroke="var(--grid)"/>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
        f'aria-label="Per-frame similarity, with the averaged score and match threshold">'
        f"{ticks}{axis}{''.join(bars)}"
        f'{ref_line(threshold, "6,4", "threshold", "var(--muted)")}'
        f'{ref_line(avg_similarity, "2,2", avg_label, "var(--text-primary)")}'
        f"</svg>"
    )


STYLE_BLOCK = f"""
:root {{
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --muted: #898781; --grid: #e1e0d9;
  --baseline: #c3c2b7; --series-1: {BLUE}; --status-good: {STATUS_GOOD};
  --status-critical: {STATUS_CRITICAL}; --border: rgba(11,11,11,0.10);
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a;
    --baseline: #383835; --series-1: #3987e5; --status-good: #0ca30c;
    --status-critical: #e66767; --border: rgba(255,255,255,0.10);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --muted: #898781; --grid: #2c2c2a;
  --baseline: #383835; --series-1: #3987e5; --status-good: #0ca30c;
  --status-critical: #e66767; --border: rgba(255,255,255,0.10);
}}
body {{ margin:0; background:var(--page); font-family: system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--text-primary); }}
.viz-root {{ max-width: 900px; margin: 0 auto; padding: 32px 24px 64px; }}
h1 {{ font-size: 22px; margin-bottom: 4px; }}
.sub {{ color: var(--text-secondary); font-size: 14px; margin-bottom: 32px; }}
section {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 20px; }}
section h2 {{ font-size: 15px; margin: 0 0 4px; }}
section .desc {{ color: var(--text-secondary); font-size: 13px; margin-bottom: 14px; }}
.step-badge {{ display:inline-block; font-size:11px; font-weight:600; color:var(--muted);
  letter-spacing:.04em; margin-bottom:6px; }}
.thumb {{ width: 100px; height: 75px; object-fit: cover; border-radius: 6px;
  border: 1px solid var(--border); margin: 0 6px 6px 0; }}
.thumb.wide {{ width: 150px; height: 100px; }}
.thumb.crop {{ width: 90px; height: 135px; object-position: center top; }}
.strip {{ display:flex; flex-wrap:wrap; }}
.headshot {{ width: 140px; height: 140px; object-fit: cover; border-radius: 10px;
  border: 1px solid var(--border); }}
.inputs-row {{ display:flex; gap: 24px; align-items:flex-start; }}
.axis-label, .ref-label {{ font-size: 10px; fill: var(--muted); font-family: inherit; }}
.chart-svg {{ width:100%; height:auto; }}
.verdict {{ display:flex; align-items:center; gap: 14px; }}
.verdict .icon {{ width:44px; height:44px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:22px; color:#fff; background: var(--status-good); }}
.verdict-fail .icon {{ background: var(--status-critical); }}
.verdict .score {{ font-size: 28px; font-weight:700; }}
.verdict .meta {{ color: var(--text-secondary); font-size:13px; }}
.stat-row {{ display:flex; gap: 28px; flex-wrap:wrap; }}
.stat {{ min-width: 120px; }}
.stat .value {{ font-size: 22px; font-weight:700; }}
.stat .label {{ color: var(--text-secondary); font-size: 12px; }}
"""


def page_shell(title, subtitle, body_html):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{STYLE_BLOCK}</style></head>
<body><div class="viz-root">
<h1>{title}</h1>
<div class="sub">{subtitle}</div>
{body_html}
</div></body></html>"""


def verdict_block(is_match, score, threshold, meta_text):
    color = "var(--status-good)" if is_match else "var(--status-critical)"
    text = "MATCH" if is_match else "NO MATCH"
    icon = "✓" if is_match else "✗"
    cls = "verdict" if is_match else "verdict verdict-fail"
    return f"""<div class="{cls}">
    <div class="icon">{icon}</div>
    <div>
      <div class="score" style="color:{color}">{text} &mdash; {score:.3f}</div>
      <div class="meta">threshold {threshold:.2f} &middot; {meta_text}</div>
    </div>
  </div>"""
