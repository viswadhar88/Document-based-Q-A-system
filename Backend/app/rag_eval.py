"""
rag_eval.py — RAG Quality Evaluation for AskYourDoc
=====================================================
Standalone script — runs independently without importing the full app.

HOW TO RUN:
-----------
  cd /mnt/d/AskYourDoc/codes/Backend
  PYTHONPATH=/mnt/d/AskYourDoc/codes/Backend python3 app/rag_eval.py

Results:
  • Printed to terminal  (block-bar style)
  • Saved to app/rag_eval_results.json
  • Chart image → app/rag_eval_chart.png   (needs matplotlib)
  • HTML report  → app/rag_eval_report.html
"""

import os
import re
import sys
import json
import time
import datetime
from pathlib import Path
from typing import List, Dict, Any

# ── Load .env manually ────────────────────────────────────────────────────────
def load_env(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

_here = Path(__file__).resolve().parent
load_env(_here.parent / ".env")
load_env(_here / ".env")

# ── DB ────────────────────────────────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("❌ psycopg2 not found. Run: pip install psycopg2-binary")
    sys.exit(1)

def get_db_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_SERVER", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "document_qa"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "viswa887"),
    )

# ── Embedding model ───────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("❌ sentence-transformers not found.")
    sys.exit(1)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
print(f"🔧 Loading embedding model: {MODEL_NAME} ...")
_model = SentenceTransformer(MODEL_NAME)
print("✅ Embedding model loaded\n")

def embed(text: str):
    return _model.encode([text], normalize_embeddings=True)[0]

# ── Config ────────────────────────────────────────────────────────────────────
TOP_K                  = 5
STRONG_MATCH_THRESHOLD = 0.55
USEFUL_THRESHOLD       = 0.40

TEST_QUERIES = [
    "Explain the main concept discussed",
    "What is the definition of the key terms used?",
    "Describe the process or method explained",
    "What are the important formulas or rules mentioned?",
    "Give an example of what is explained",
    "What are the key points covered?",
    "How does this concept work?",
    "What are the steps or stages involved?",
]

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","as",
    "into","through","and","or","but","if","then","that","this","it","its",
    "i","you","we","they","not","no","so","up","out","about","what","your",
    "also","just","more","some","any","all","each","which","how","when",
    "where","who","there","their","than","very","would",
}

def tokenize(text: str) -> set:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in STOPWORDS}

# ── Load chunks from processed text files ────────────────────────────────────
def load_chunks(conn) -> List[Dict]:
    chunks = []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, filename, file_path, file_type FROM documents")
        docs = cur.fetchall()

    processed_dir = _here.parent / "data" / "processed"

    for doc in docs:
        stem = Path(doc["file_path"]).stem if doc["file_path"] else doc["filename"]
        ext  = (doc["file_type"] or "pdf").lstrip(".")
        txt  = processed_dir / f"{stem}_{ext}.txt"
        if not txt.exists():
            continue
        text = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        chunk_size, overlap = 1000, 200
        start, idx = 0, 0
        while start < len(text):
            end   = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append({
                    "doc_id": str(doc["id"]),
                    "filename": doc["filename"],
                    "idx": idx,
                    "text": chunk,
                })
                idx += 1
            start += chunk_size - overlap
    return chunks

_chunks_cache = []
_emb_cache    = None

def get_chunks_emb(conn):
    global _chunks_cache, _emb_cache
    if _chunks_cache:
        return _chunks_cache, _emb_cache
    print("  📚 Embedding document chunks (one-time)...")
    _chunks_cache = load_chunks(conn)
    if not _chunks_cache:
        return [], None
    texts    = [c["text"] for c in _chunks_cache]
    _emb_cache = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    print(f"  ✅ {len(_chunks_cache)} chunks ready\n")
    return _chunks_cache, _emb_cache

def search(query, chunks, embs, top_k=TOP_K):
    if not chunks or embs is None:
        return []
    q   = embed(query)
    sim = embs @ q
    idx = np.argsort(sim)[::-1][:top_k]
    return [{**chunks[i], "score": float(sim[i])} for i in idx]

# ── Result helper ─────────────────────────────────────────────────────────────
def R(metric, score, display, detail):
    return {"metric": metric, "score": score, "display": display, "detail": detail}

# ── METRIC 1 — Index Coverage ─────────────────────────────────────────────────
def m1_index_coverage(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT COUNT(*) AS t FROM documents")
        total = cur.fetchone()["t"]
        cur.execute("SELECT COUNT(*) AS i FROM documents WHERE chunks_created > 0")
        indexed = cur.fetchone()["i"]
    if total == 0:
        return R("Index Coverage", 0, "0.0%", "No documents uploaded yet")
    pct = (indexed / total) * 100
    return R("Index Coverage", round(pct,1), f"{pct:.1f}%",
             f"{indexed}/{total} documents chunked & indexed")

# ── METRIC 2 — Embedding Hit Rate ────────────────────────────────────────────
def m2_hit_rate(chunks, embs):
    if not chunks:
        return R("Embedding Hit Rate", 0, "N/A", "No chunks available")
    hits, total, tops = 0, 0, []
    for q in TEST_QUERIES:
        res = search(q, chunks, embs)
        if not res:
            total += 1
            continue
        top = res[0]["score"]
        tops.append(top)
        if top >= STRONG_MATCH_THRESHOLD:
            hits += 1
        total += 1
    rate    = (hits/total*100) if total else 0
    avg_top = sum(tops)/len(tops) if tops else 0
    return R("Embedding Hit Rate", round(rate,1), f"{rate:.1f}%",
             f"{hits}/{total} queries had strong match (≥{STRONG_MATCH_THRESHOLD}) | avg top: {avg_top:.3f}")

# ── METRIC 3 — Chunk Retrieval Quality ───────────────────────────────────────
def m3_retrieval_quality(chunks, embs):
    if not chunks:
        return R("Chunk Retrieval Quality", 0, "0.000", "No chunks available")
    all_s = []
    for q in TEST_QUERIES:
        for r in search(q, chunks, embs):
            all_s.append(r["score"])
    if not all_s:
        return R("Chunk Retrieval Quality", 0, "0.000", "No results")
    avg = sum(all_s)/len(all_s)
    mn, mx = min(all_s), max(all_s)
    rating = "Excellent" if avg>=0.65 else "Good" if avg>=0.50 else "Fair" if avg>=0.35 else "Poor"
    return R("Chunk Retrieval Quality", round(avg,3), f"{avg:.3f} ({rating})",
             f"Avg:{avg:.3f} Min:{mn:.3f} Max:{mx:.3f} | {len(all_s)} chunks scored")

# ── METRIC 4 — Context Utilization ───────────────────────────────────────────
def m4_utilization(chunks, embs):
    if not chunks:
        return R("Context Utilization", 0, "0.0%", "No chunks available")
    ratios, u_total, r_total = [], 0, 0
    for q in TEST_QUERIES:
        res = search(q, chunks, embs)
        if not res:
            continue
        useful = sum(1 for r in res if r["score"] >= USEFUL_THRESHOLD)
        ratios.append(useful / len(res))
        u_total += useful
        r_total += len(res)
    if not ratios:
        return R("Context Utilization", 0, "0.0%", "No results")
    avg = (sum(ratios)/len(ratios))*100
    return R("Context Utilization", round(avg,1), f"{avg:.1f}%",
             f"{u_total} useful out of {r_total} retrieved (threshold ≥{USEFUL_THRESHOLD})")

# ── METRIC 5 — Query Success Rate ────────────────────────────────────────────
def m5_success_rate(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT COUNT(*) AS t FROM query_history")
        total = cur.fetchone()["t"]
        if total == 0:
            return R("Query Success Rate", 0, "N/A", "No query history yet — ask some questions first")
        cur.execute("SELECT COUNT(*) AS s FROM query_history WHERE success='true'")
        success = cur.fetchone()["s"]
        cur.execute("SELECT success FROM query_history ORDER BY created_at DESC LIMIT 10")
        recent    = cur.fetchall()
        recent_ok = sum(1 for r in recent if r["success"]=="true")
    rate = (success/total)*100
    rr   = (recent_ok/len(recent)*100) if recent else 0
    return R("Query Success Rate", round(rate,1), f"{rate:.1f}%",
             f"{success}/{total} succeeded | Last 10: {recent_ok}/10 ({rr:.0f}%)")



# ── Tier System ───────────────────────────────────────────────────────────────
TIERS = {
    "Index Coverage":          [(95,"S"),(80,"A"),(60,"B"),(0,"C")],
    "Embedding Hit Rate":      [(75,"S"),(55,"A"),(35,"B"),(0,"C")],
    "Chunk Retrieval Quality": [(0.65,"S"),(0.55,"A"),(0.45,"B"),(0,"C")],
    "Context Utilization":     [(80,"S"),(60,"A"),(40,"B"),(0,"C")],
    "Query Success Rate":      [(90,"S"),(75,"A"),(55,"B"),(0,"C")],
}
ICONS  = {"S":"🏆","A":"✅","B":"⚠️ ","C":"❌"}
COLORS = {"S":"\033[92m","A":"\033[96m","B":"\033[93m","C":"\033[91m"}  # green/cyan/yellow/red
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def get_tier(metric, score):
    for t, label in TIERS.get(metric, []):
        if score >= t:
            return label
    return "C"

def to_pct(metric, score):
    """Normalize any score to 0-100 for bar display."""
    if metric == "Chunk Retrieval Quality":
        return score * 100
    return score

# ── Animated block bar ────────────────────────────────────────────────────────
def bbar(pct, w=30, tier="S"):
    """
    Renders a block-style progress bar.
    Full blocks: █  Partial hint: ▓  Empty: ░
    """
    filled  = min(int((pct / 100.0) * w), w)
    partial = 1 if filled < w and (pct / 100.0 * w - filled) >= 0.5 else 0
    empty   = w - filled - partial
    bar     = "█" * filled + ("▓" if partial else "") + "░" * empty
    color   = COLORS.get(tier, "")
    return f"{color}[{bar}]{RESET}"

# ── Animated terminal print (simulates fill) ──────────────────────────────────
def print_bar_animated(label, pct, tier, display, detail, animate=True):
    color = COLORS[tier]
    w     = 30

    if animate:
        sys.stdout.write(f"\n  {BOLD}{label}{RESET}\n  ")
        for step in range(1, w + 1):
            step_pct = (step / w) * 100
            if step_pct > pct:
                step_pct = pct
            filled  = int((step_pct / 100) * w)
            partial = 1 if filled < w else 0
            empty   = w - filled - partial
            bar     = "█" * filled + ("▓" if partial else "") + "░" * empty
            sys.stdout.write(f"\r  {color}[{bar}]{RESET}  {display}")
            sys.stdout.flush()
            time.sleep(0.018)
        print()
    else:
        print(f"\n  {BOLD}{label}{RESET}")
        print(f"  {bbar(pct, w, tier)}  {display}")

    print(f"  {DIM}→ {detail}{RESET}")

# ── Terminal Report ───────────────────────────────────────────────────────────
def print_report(results, animate=True):
    W = 72
    print("\n" + "═" * W)
    print(f"  {BOLD}AskYourDoc — RAG Quality Evaluation Report{RESET}")
    print("═" * W)

    tc = {"S":0,"A":0,"B":0,"C":0}
    for r in results:
        n, s, d, det = r["metric"], r["score"], r["display"], r["detail"]
        t   = get_tier(n, s)
        pct = to_pct(n, s)
        tc[t] += 1
        print_bar_animated(n, pct, t, d, det, animate=animate)

    print("\n" + "─" * W)

    if tc["C"]==0 and tc["B"]<=1:
        verdict = "RAG system performing at a high level"
    elif tc["C"]==0:
        verdict = "Good overall — a few areas to improve"
    elif tc["C"]<=2:
        verdict = "Fair — some pipeline areas need attention"
    else:
        verdict = "Several metrics need work"
    print(f"\n  {BOLD}{verdict}{RESET}")
    print("═" * W + "\n")

    # Save JSON
    out = _here / "rag_eval_results.json"
    out.write_text(json.dumps({"metrics": results, "tiers": tc,
                               "generated_at": datetime.datetime.now().isoformat()},
                              indent=2, default=str))
    print(f"  📄 JSON saved → {out}")
    return tc

# ── Matplotlib Charts ─────────────────────────────────────────────────────────
def save_charts_png(results, tc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping chart (pip install matplotlib)")
        return

    TIER_HEX = {"S":"#4ade80","A":"#38bdf8","B":"#facc15","C":"#f87171"}

    labels  = [r["metric"].replace(" ", "\n") for r in results]
    scores  = [to_pct(r["metric"], r["score"]) for r in results]
    tiers   = [get_tier(r["metric"], r["score"]) for r in results]
    bar_clr = [TIER_HEX[t] for t in tiers]

    fig = plt.figure(figsize=(16, 9), facecolor="#0f1117")
    fig.suptitle("AskYourDoc — RAG Quality Evaluation",
                 color="#e2e8f0", fontsize=18, fontweight="bold", y=0.97)

    # ── Panel 1: Horizontal bar chart ─────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, facecolor="#1e2130")
    bars = ax1.barh(labels[::-1], scores[::-1], color=bar_clr[::-1],
                    height=0.55, edgecolor="#2d3148", linewidth=0.8)

    for bar, score, tier in zip(bars, scores[::-1], tiers[::-1]):
        ax1.text(min(score + 1.5, 97), bar.get_y() + bar.get_height()/2,
                 f"{score:.1f}%  [{tier}]", va="center", ha="left",
                 color=TIER_HEX[tier], fontsize=9, fontweight="bold")

    ax1.set_xlim(0, 115)
    ax1.set_xlabel("Score (%)", color="#94a3b8", fontsize=10)
    ax1.set_title("Metric Scores", color="#e2e8f0", fontsize=13, pad=12)
    ax1.tick_params(colors="#94a3b8", labelsize=8)
    ax1.spines[:].set_color("#2d3148")
    ax1.set_facecolor("#1e2130")
    ax1.axvline(100, color="#2d3148", linestyle="--", linewidth=0.8)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#2d3148")

    # ── Panel 2: Radar chart ──────────────────────────────────────────────────
    N       = len(results)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals    = [min(to_pct(r["metric"], r["score"]), 100) / 100 for r in results]
    vals   += vals[:1]
    short   = [r["metric"].replace(" Quality","").replace(" Rate","").replace(" Coverage","")
               for r in results]

    ax2 = fig.add_subplot(1, 2, 2, polar=True, facecolor="#1e2130")
    ax2.set_facecolor("#1e2130")
    ax2.plot(angles, vals, "o-", linewidth=2, color="#818cf8")
    ax2.fill(angles, vals, alpha=0.25, color="#818cf8")
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(short, color="#94a3b8", size=8)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_tick_params(labelcolor="#475569", labelsize=7)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["25%","50%","75%","100%"])
    ax2.grid(color="#2d3148", linewidth=0.8)
    ax2.spines["polar"].set_color("#2d3148")
    ax2.set_title("Radar Overview", color="#e2e8f0", fontsize=13, pad=20)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=TIER_HEX["S"], label="S Tier — Excellent"),
        mpatches.Patch(color=TIER_HEX["A"], label="A Tier — Good"),
        mpatches.Patch(color=TIER_HEX["B"], label="B Tier — Fair"),
        mpatches.Patch(color=TIER_HEX["C"], label="C Tier — Poor"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               facecolor="#1e2130", edgecolor="#2d3148",
               labelcolor="#94a3b8", fontsize=9, framealpha=1,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = _here / "rag_eval_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Chart saved  → {out}")

# ── HTML Report ───────────────────────────────────────────────────────────────
def save_html_report(results, tc):
    TIER_CSS = {"S":"#4ade80","A":"#38bdf8","B":"#facc15","C":"#f87171"}
    TIER_BG  = {"S":"rgba(74,222,128,0.12)","A":"rgba(56,189,248,0.12)",
                "B":"rgba(250,204,21,0.12)","C":"rgba(248,113,113,0.12)"}
    TIER_LABEL = {"S":"S Tier","A":"A Tier","B":"B Tier","C":"C Tier"}
    ICON = {"S":"🏆","A":"✅","B":"⚠️","C":"❌"}

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build radar data for Chart.js
    radar_labels  = json.dumps([r["metric"] for r in results])
    radar_values  = json.dumps([min(to_pct(r["metric"], r["score"]), 100) for r in results])
    bar_labels    = json.dumps([r["metric"] for r in results])
    bar_values    = json.dumps([min(to_pct(r["metric"], r["score"]), 100) for r in results])
    bar_colors    = json.dumps([TIER_CSS[get_tier(r["metric"], r["score"])] for r in results])

    # Build metric cards HTML
    cards_html = ""
    for r in results:
        n, s, d, det = r["metric"], r["score"], r["display"], r["detail"]
        t    = get_tier(n, s)
        pct  = min(to_pct(n, s), 100)
        col  = TIER_CSS[t]
        bg   = TIER_BG[t]
        icon = ICON[t]
        tlbl = TIER_LABEL[t]
        cards_html += f"""
        <div class="card" style="--accent:{col};--bg:{bg};">
          <div class="card-header">
            <span class="metric-name">{n}</span>
            <span class="tier-badge" style="color:{col};border-color:{col};background:{bg};">
              {icon} {tlbl}
            </span>
          </div>
          <div class="bar-wrap">
            <div class="bar-track">
              <div class="bar-fill" style="width:0%;background:{col};"
                   data-target="{pct:.1f}"></div>
            </div>
            <span class="bar-label" style="color:{col};">{d}</span>
          </div>
          <p class="detail">→ {det}</p>
        </div>
        """

    tier_summary = (
        f"<span class='ts s'>🏆 S = {tc['S']}</span>"
        f"<span class='ts a'>✅ A = {tc['A']}</span>"
        f"<span class='ts b'>⚠️  B = {tc['B']}</span>"
        f"<span class='ts c'>❌ C = {tc['C']}</span>"
    )

    if tc["C"]==0 and tc["B"]<=1:
        verdict = "🏆 RAG pipeline performing at a high level"
        vcls    = "verdict-s"
    elif tc["C"]==0:
        verdict = "✅ Good overall — a few areas to improve"
        vcls    = "verdict-a"
    elif tc["C"]<=2:
        verdict = "⚠️ Fair — some pipeline areas need attention"
        vcls    = "verdict-b"
    else:
        verdict = "❌ Several metrics need work"
        vcls    = "verdict-c"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AskYourDoc — RAG Eval Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:       #0a0d14;
    --surface:  #111827;
    --card:     #161d2e;
    --border:   #1e293b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --s:        #4ade80;
    --a:        #38bdf8;
    --b:        #facc15;
    --c:        #f87171;
  }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem 1rem;
  }}

  .container {{ max-width: 960px; margin: 0 auto; }}

  /* Header */
  header {{
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    background: linear-gradient(135deg, #818cf8, #38bdf8, #4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
  }}
  header p {{ color: var(--muted); font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; }}

  /* Verdict */
  .verdict {{
    text-align: center;
    padding: 0.75rem 1.5rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1.8rem;
    border: 1px solid;
  }}
  .verdict-s {{ color: var(--s); border-color: var(--s); background: rgba(74,222,128,0.08); }}
  .verdict-a {{ color: var(--a); border-color: var(--a); background: rgba(56,189,248,0.08); }}
  .verdict-b {{ color: var(--b); border-color: var(--b); background: rgba(250,204,21,0.08); }}
  .verdict-c {{ color: var(--c); border-color: var(--c); background: rgba(248,113,113,0.08); }}

  /* Tier summary */
  .tier-summary {{
    display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;
    margin-bottom: 2rem;
  }}
  .ts {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; font-weight: 600;
    padding: 0.3rem 0.8rem; border-radius: 6px;
    border: 1px solid;
  }}
  .ts.s {{ color: var(--s); border-color: var(--s); background: rgba(74,222,128,0.1); }}
  .ts.a {{ color: var(--a); border-color: var(--a); background: rgba(56,189,248,0.1); }}
  .ts.b {{ color: var(--b); border-color: var(--b); background: rgba(250,204,21,0.1); }}
  .ts.c {{ color: var(--c); border-color: var(--c); background: rgba(248,113,113,0.1); }}

  /* Metric cards */
  .cards {{ display: grid; gap: 1rem; }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    transition: transform 0.2s, box-shadow 0.2s;
    background: linear-gradient(135deg, var(--bg) 0%, var(--bg) 70%);
  }}
  .card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
  }}
  .card-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;
  }}
  .metric-name {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem; font-weight: 700; color: var(--text);
  }}
  .tier-badge {{
    font-size: 0.72rem; font-weight: 700;
    padding: 0.2rem 0.65rem; border-radius: 20px; border: 1px solid;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;
  }}

  /* Progress bar */
  .bar-wrap {{
    display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.55rem;
  }}
  .bar-track {{
    flex: 1; height: 10px; background: var(--border);
    border-radius: 999px; overflow: hidden;
  }}
  .bar-fill {{
    height: 100%; border-radius: 999px;
    transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
    box-shadow: 0 0 10px currentColor;
  }}
  .bar-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; font-weight: 700; white-space: nowrap;
  }}
  .detail {{
    color: var(--muted); font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
    line-height: 1.5;
  }}

  /* Charts section */
  .charts-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem; font-weight: 700;
    color: var(--muted); letter-spacing: 0.08em;
    text-transform: uppercase; margin: 2.5rem 0 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
  }}
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    margin-bottom: 2rem;
  }}
  @media(max-width: 640px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
  .chart-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
  }}
  .chart-box h3 {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 1rem;
  }}

  /* Footer */
  footer {{
    text-align: center; color: var(--muted);
    font-size: 0.72rem; margin-top: 2rem;
    font-family: 'JetBrains Mono', monospace;
    padding-top: 1rem; border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>AskYourDoc — RAG Quality Evaluation</h1>
    <p>Generated: {now}</p>
  </header>

  <div class="verdict {vcls}">{verdict}</div>

  <div class="tier-summary">{tier_summary}</div>

  <div class="cards">
    {cards_html}
  </div>

  <p class="charts-title">Visual Breakdown</p>
  <div class="charts-grid">
    <div class="chart-box">
      <h3>Metric Scores — Horizontal Bar</h3>
      <canvas id="barChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Radar Overview</h3>
      <canvas id="radarChart"></canvas>
    </div>
  </div>

  <footer>AskYourDoc RAG Eval • {now}</footer>
</div>

<script>
  // Animate progress bars after load
  window.addEventListener('load', () => {{
    document.querySelectorAll('.bar-fill').forEach(el => {{
      const target = el.dataset.target;
      setTimeout(() => {{ el.style.width = target + '%'; }}, 150);
    }});
  }});

  const LABELS  = {radar_labels};
  const VALUES  = {radar_values};
  const BCOLORS = {bar_colors};

  const gridColor  = 'rgba(30,41,59,0.8)';
  const tickColor  = '#64748b';
  const legendClr  = '#94a3b8';

  // Bar chart
  new Chart(document.getElementById('barChart'), {{
    type: 'bar',
    data: {{
      labels: LABELS,
      datasets: [{{
        label: 'Score (%)',
        data: VALUES,
        backgroundColor: BCOLORS.map(c => c + 'cc'),
        borderColor: BCOLORS,
        borderWidth: 1.5,
        borderRadius: 5,
      }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      animation: {{ duration: 1200, easing: 'easeOutQuart' }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#1e2130',
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          borderColor: '#2d3148',
          borderWidth: 1,
        }}
      }},
      scales: {{
        x: {{
          min: 0, max: 110,
          grid: {{ color: gridColor }},
          ticks: {{ color: tickColor, font: {{ family: 'JetBrains Mono', size: 10 }} }},
        }},
        y: {{
          grid: {{ display: false }},
          ticks: {{ color: tickColor, font: {{ family: 'JetBrains Mono', size: 9 }} }},
        }}
      }}
    }}
  }});

  // Radar chart
  new Chart(document.getElementById('radarChart'), {{
    type: 'radar',
    data: {{
      labels: LABELS.map(l => l.replace(' Quality','').replace(' Rate','').replace(' Coverage','')),
      datasets: [{{
        label: 'Score',
        data: VALUES,
        backgroundColor: 'rgba(129,140,248,0.2)',
        borderColor: '#818cf8',
        borderWidth: 2,
        pointBackgroundColor: BCOLORS,
        pointBorderColor: '#0a0d14',
        pointRadius: 5,
      }}]
    }},
    options: {{
      responsive: true,
      animation: {{ duration: 1400, easing: 'easeOutQuart' }},
      scales: {{
        r: {{
          min: 0, max: 100,
          grid: {{ color: gridColor }},
          angleLines: {{ color: gridColor }},
          ticks: {{
            color: tickColor, backdropColor: 'transparent',
            font: {{ family: 'JetBrains Mono', size: 9 }}, stepSize: 25,
          }},
          pointLabels: {{ color: legendClr, font: {{ family: 'JetBrains Mono', size: 9 }} }},
        }}
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#1e2130',
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          borderColor: '#2d3148',
          borderWidth: 1,
        }}
      }}
    }}
  }});
</script>
</body>
</html>
"""

    out = _here / "rag_eval_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"  🌐 HTML report → {out}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🔍 AskYourDoc RAG Evaluation\n")
    try:
        conn = get_db_conn()
        print("✅ Connected to PostgreSQL\n")
    except Exception as e:
        print(f"❌ DB connection failed: {e}")
        sys.exit(1)

    chunks, embs = get_chunks_emb(conn)
    results = []

    print("📊 Running 5 metrics...\n")
    print("  [1/5] Index Coverage...")
    results.append(m1_index_coverage(conn))
    print("  [2/5] Embedding Hit Rate...")
    results.append(m2_hit_rate(chunks, embs))
    print("  [3/5] Chunk Retrieval Quality...")
    results.append(m3_retrieval_quality(chunks, embs))
    print("  [4/5] Context Utilization...")
    results.append(m4_utilization(chunks, embs))
    print("  [5/5] Query Success Rate...")
    results.append(m5_success_rate(conn))

    conn.close()

    print("\n")
    tc = print_report(results, animate=True)

    print("\n  💾 Saving outputs...")
    save_charts_png(results, tc)
    save_html_report(results, tc)
    print()

if __name__ == "__main__":
    main()
