"""FPL captaincy — eval dashboard (read-only).

A visualization of the eval results. It reads ONLY the committed JSON artifacts in
`data/eval_results/` — it never calls the model and needs no ANTHROPIC_API_KEY, so it
is free to run and safe to deploy as a public link. To refresh the data, re-run the
evals (`python src/evals/decision_quality.py` and `retrieval_eval.py`), which rewrite
those JSON files; this app only renders what they saved.

Run locally:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "eval_results"

# palette mirrors assets/make_charts.py so the dashboard and the README charts match
INK = "#1d2433"
RAG_C = "#2563eb"      # blue — the RAG pick
CROWD_C = "#9ca3af"    # grey — the crowd template
WIN_C = "#16a34a"      # green
BUST_C = "#dc2626"     # red
HIGH_C = "#1e3a8a"     # dark blue — high-confidence bucket
MED_C = "#3b82f6"      # lighter blue — medium-confidence bucket


@st.cache_data
def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:#fff;border-radius:6px;"
        f"padding:2px 8px;font-size:0.8rem;font-weight:600'>{text}</span>"
    )


def result_tag(row: dict) -> tuple[str, str]:
    """(label, color) for a per-GW result. Bust is flagged on top of win/tie/loss."""
    base = {"win": ("win", WIN_C), "tie": ("tie", CROWD_C), "loss": ("loss", BUST_C)}
    label, color = base.get(row["result"], (row["result"], CROWD_C))
    if row.get("bust"):
        return (f"{label} · bust", BUST_C)
    return (label, color)


# ----------------------------------------------------------------------------------
st.set_page_config(page_title="FPL captaincy — eval dashboard", page_icon="⚽", layout="wide")

dq = load("decision_quality.json")
rt = load("retrieval.json")
agg = dq["aggregate"]
rec = agg["record"]
n = dq["meta"]["n"]

# ---- header -----------------------------------------------------------------------
st.title("FPL captaincy — eval dashboard")
st.markdown(
    "**Does retrieved context lead to a better captaincy decision than the crowd — "
    "and can you trust the system's confidence?** Graded on real 2025–26 outcomes."
)
st.caption(
    f"Read-only view of saved eval output (n={n} gameweeks, K={dq['meta']['k']} votes "
    "each). Directional only — no statistical claim. No model is called here."
)
st.divider()

# ---- metric cards -----------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Divergence differential",
    f"+{agg['differential']} pts",
    delta=f"{len(agg['divergent_gws'])} divergent GWs",
    delta_color="off",
)
c2.metric("Record vs crowd", f"{rec['w']}W–{rec['l']}L–{rec['t']}T")
c3.metric("Agreement rate", f"{agg['agreement_rate']:.0%}", help="GWs RAG echoed the crowd template")
c4.metric(
    "Total captain pts (RAG)",
    agg["rag_total"],
    delta=f"+{agg['rag_total'] - agg['template_total']} vs template ({agg['template_total']})",
)
st.caption(
    "The headline is the **divergence-only differential**: RAG is scored only on weeks "
    "its pick differs from the crowd. On weeks it agrees it adds nothing by definition, "
    "so counting those would hand it free credit for obvious picks."
)
st.divider()

# ---- HEADLINE: confidence calibration ---------------------------------------------
st.header("Can you trust its confidence? — No.")
st.markdown(
    "Its **most-confident picks did no better** (slightly worse) than its least-confident "
    "ones. The edge lives in the calls it is *unsure* about. This is the project's "
    "headline finding."
)

buckets = dq["calibration"]["capture_by_bucket"]
cap_rows = [
    {"bucket": f"{b}\n(n={buckets[b]['n']})", "capture": buckets[b]["mean_capture"], "key": b}
    for b in ("high", "medium", "low")
    if buckets[b]["n"] and buckets[b]["mean_capture"] is not None
]
cap_df = pd.DataFrame(cap_rows)

left, right = st.columns([3, 2])
with left:
    bar = (
        alt.Chart(cap_df)
        .mark_bar(size=70)
        .encode(
            x=alt.X("bucket:N", title="confidence bucket", sort=["high", "medium", "low"],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("capture:Q", title="mean normalized capture (floor 0 → ceiling 1)",
                    scale=alt.Scale(domain=[0, 0.7])),
            color=alt.Color("key:N", scale=alt.Scale(domain=["high", "medium"],
                            range=[HIGH_C, MED_C]), legend=None),
        )
    )
    labels = bar.mark_text(dy=-10, fontSize=15, fontWeight="bold", color=INK).encode(
        text=alt.Text("capture:Q", format=".2f")
    )
    st.altair_chart((bar + labels).properties(height=360), width='stretch')
with right:
    hi, me = buckets["high"], buckets["medium"]
    st.metric("High-confidence capture", f"{hi['mean_capture']:.2f}",
              delta=f"{hi['mean_capture'] - me['mean_capture']:+.2f} vs medium",
              delta_color="inverse")
    st.markdown(
        f"High-confidence picks captured **{hi['mean_capture']:.2f}** of available value "
        f"versus **{me['mean_capture']:.2f}** for the medium ones — flat, even inverted."
    )
    st.markdown(
        "**Why it's mechanical:** every high-confidence week was a *captain-the-obvious-"
        "premium* agreement with the crowd, and several of those obvious picks simply "
        "failed. The model is confident when the choice is **obvious**, not when it's right."
    )

# confident busts table, straight from the per-GW data
busts = [r for r in dq["per_gw"] if r.get("bust")]
if busts:
    st.markdown("**Confident template busts** (captain returned ≤ "
                f"{dq['meta']['bust_pts']} pts — no goal/assist):")
    st.dataframe(
        pd.DataFrame([
            {"GW": r["gw"], "pick": r["rag_pick"], "pts": r["rag_pts"],
             "confidence": r["confidence_mean"], "bucket": r["confidence_bucket"]}
            for r in busts
        ]),
        hide_index=True, width='content',
    )
st.caption(
    "Capture is an absolute quality signal (random floor = 0, perfect hindsight = 1), "
    "comparable across buckets unlike the vs-template delta. Confound, stated plainly: "
    "high-confidence GWs are largely template agreements, so their vs-template delta is "
    f"~0 by construction. n={n} — read as directional."
)
st.divider()

# ---- where it beats the crowd -----------------------------------------------------
st.header("Where it beats the crowd")
div_rows = [r for r in dq["per_gw"] if r["divergence"]]
long = []
for r in div_rows:
    long.append({"gw": f"GW{r['gw']}", "who": "RAG pick", "pts": r["rag_pts"],
                 "label": r["rag_pick"]})
    long.append({"gw": f"GW{r['gw']}", "who": "Crowd (most-captained)", "pts": r["crowd_pts"],
                 "label": r["crowd_pick"]})
div_df = pd.DataFrame(long)

dl, dr = st.columns([3, 2])
with dl:
    base = alt.Chart(div_df).encode(
        x=alt.X("gw:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("who:N", sort=["RAG pick", "Crowd (most-captained)"]),
    )
    bars = base.mark_bar().encode(
        y=alt.Y("pts:Q", title="captain points (real 2025–26 outcome)"),
        color=alt.Color("who:N", scale=alt.Scale(
            domain=["RAG pick", "Crowd (most-captained)"], range=[RAG_C, CROWD_C]),
            legend=alt.Legend(title=None, orient="top")),
    )
    txt = base.mark_text(dy=-6, fontSize=11, color=INK).encode(
        y="pts:Q", text="label:N",
        detail="who:N",
    )
    st.altair_chart((bars + txt).properties(height=360), width='stretch')
with dr:
    for r in div_rows:
        st.markdown(
            f"**GW{r['gw']}** · {r['rag_pick']} **{r['rag_pts']}** vs "
            f"{r['crowd_pick']} {r['crowd_pts']} "
            f"{badge(f'+{r['delta']}', WIN_C)} "
            f"<span style='color:{CROWD_C}'>({r['confidence_bucket']} conf)</span>",
            unsafe_allow_html=True,
        )
    st.markdown(f"**Net: +{agg['differential']} pts over {len(div_rows)} divergent GWs.**")
st.caption(
    "Every divergence win is *medium* confidence and the model flagged each a close call "
    "— directional evidence, not a confident verdict. RAG beats the crowd only where it "
    "diverges, and it diverges only on calls it is itself unsure about."
)
st.divider()

# ---- retrieval health -------------------------------------------------------------
st.header("Retrieval health")
st.markdown(
    "Are the right notes fetched? The one interactive control: flip the temporal filter "
    "**off** to watch the scores drop — future-dated notes leak in and outrank the "
    "legitimate earlier note."
)
filter_on = st.toggle("Apply temporal filter (date < GW deadline)", value=True)
mode = "on" if filter_on else "off"
ret = rt["aggregate"][mode]
ref = rt["aggregate"]["on"]

rc1, rc2, rc3 = st.columns(3)
rc1.metric(f"hit@{rt['meta']['k']}", f"{ret['hit_at_k']:.3f}",
           delta=None if filter_on else f"{ret['hit_at_k'] - ref['hit_at_k']:+.3f} vs ON",
           delta_color="normal")
rc2.metric(f"recall@{rt['meta']['k']}", f"{ret['recall_at_k']:.3f}",
           delta=None if filter_on else f"{ret['recall_at_k'] - ref['recall_at_k']:+.3f} vs ON",
           delta_color="normal")
passed = rt["temporal_assertion"]["passed"]
with rc3:
    st.markdown("**No-leak assertion**", help="Hard assertion: no retrieved note dated "
                "on/after its case deadline.")
    st.markdown(
        badge("PASS — no hindsight leak", WIN_C) if passed
        else badge(f"FAIL — {len(rt['temporal_assertion']['violations'])} leak(s)", BUST_C),
        unsafe_allow_html=True,
    )

if filter_on:
    st.caption(
        f"Counter-intuitively the filter *raises* the aggregate (OFF: hit "
        f"{rt['aggregate']['off']['hit_at_k']:.3f} / recall "
        f"{rt['aggregate']['off']['recall_at_k']:.3f}) — the future notes had been "
        "outranking the legitimate earlier note. The temporal claim, demonstrated."
    )
else:
    st.caption(
        f"Filter OFF: scores **drop** to hit {ret['hit_at_k']:.3f} / recall "
        f"{ret['recall_at_k']:.3f} (ON: {ref['hit_at_k']:.3f} / {ref['recall_at_k']:.3f}). "
        f"Across {rt['meta']['n_cases']} cases, hindsight leakage costs accuracy."
    )

with st.expander("Per-case retrieved doc_ids"):
    st.dataframe(
        pd.DataFrame([
            {"GW": c["gw"], "question": c["question"],
             "relevant": ", ".join(c["relevant_doc_ids"]),
             f"hit ({mode})": c[mode]["hit"], f"recall ({mode})": c[mode]["recall"],
             "retrieved": ", ".join(c[mode]["doc_ids"])}
            for c in rt["per_case"]
        ]),
        hide_index=True, width='stretch',
    )
st.divider()

# ---- per-gameweek detail ----------------------------------------------------------
st.header("Per-gameweek detail")
st.caption("All 8 graded gameweeks. Each row is read-only saved data — the candidates, "
           "the case for each, and the mean confidence the system produced that week.")

strip = st.columns(len(dq["per_gw"]))
for col, r in zip(strip, dq["per_gw"]):
    label, color = result_tag(r)
    col.markdown(
        f"<div style='text-align:center'><b>GW{r['gw']}</b><br>"
        f"<span style='font-size:1.4rem;font-weight:700;color:{INK}'>{r['rag_pts']}</span>"
        f"<br>{badge(label, color)}</div>",
        unsafe_allow_html=True,
    )

st.write("")
for r in dq["per_gw"]:
    label, color = result_tag(r)
    header = (f"GW{r['gw']} — {r['rag_pick']} {r['rag_pts']} pts · {label} · "
              f"{r['confidence_bucket']} confidence ({r['confidence_mean']:.2f} mean)")
    with st.expander(header):
        st.markdown(f"*{r['question']}*")
        cols = st.columns(3)
        cols[0].metric("RAG pick", f"{r['rag_pick']}", delta=f"{r['rag_pts']} pts",
                       delta_color="off")
        cols[1].metric("Crowd pick", f"{r['crowd_pick']}", delta=f"{r['crowd_pts']} pts",
                       delta_color="off")
        cols[2].metric("Ceiling (hindsight)", f"{r['ceiling_pick']}",
                       delta=f"{r['ceiling_pts']} pts", delta_color="off")
        dec = r.get("decision")
        if dec:
            st.markdown(f"**Lean:** {dec['forced_pick_name']} — {dec['rationale']}")
            st.markdown("**Candidates the notes supported:**")
            for cand in dec["candidates"]:
                st.markdown(f"- **{cand['name']}** — {cand['case']}")
        st.caption(
            f"Confidence shown is the **mean over {dq['meta']['k']} runs** (the value the "
            "bucketing uses); the reasoning text is from one representative run that "
            "produced the majority pick. Vote share "
            f"{r['vote_share']:.0%} · flagged close in {r['is_close_rate']:.0%} of runs."
        )

st.divider()
st.caption(
    f"All numbers are read from committed eval output (n={n}, directional). The "
    "answer-quality LLM-judge scores are intentionally omitted here — they carry a "
    "same-model-class leniency caveat and are the project's most-caveated number. "
    "Source: `data/eval_results/*.json`, regenerated by the eval scripts."
)
