import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px

import db
from engine import evaluate_trace, run_mock_agent, run_http_agent, run_anthropic_agent
from evaluators.cost_latency import estimate_cost, PRICING_PER_1K

st.set_page_config(page_title="AI Agent Evaluation & Reliability Platform", layout="wide")
db.init_db()

# ---------------- Sidebar: judge / agent config ----------------
st.sidebar.title("⚙️ Configuration")

anthropic_key = st.sidebar.text_input("Anthropic API key (optional)", type="password",
                                       help="Enables LLM-judge scoring and the built-in Anthropic agent runner. "
                                            "Without it, the platform uses fast rule-based evaluators.")
judge_model = st.sidebar.selectbox("Model for LLM judge / agent runner",
                                    list(PRICING_PER_1K.keys()), index=0)
use_llm_judge = st.sidebar.checkbox("Use LLM-as-judge for correctness & faithfulness", value=False,
                                     disabled=not bool(anthropic_key))

judge_client = None
if anthropic_key:
    try:
        import anthropic
        judge_client = anthropic.Anthropic(api_key=anthropic_key)
    except Exception as e:
        st.sidebar.error(f"Could not init Anthropic client: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("All data is stored locally in SQLite (`data/eval_platform.db`). "
                    "No data leaves this app except optional LLM-judge calls to Anthropic.")

# ---------------- Header ----------------
st.title("🧪 AI Agent Evaluation & Reliability Platform")
st.caption("Evaluate correctness, faithfulness, RAG quality, tool-call accuracy, cost, latency, "
           "and safety across agent versions — and catch regressions before they ship.")

tabs = st.tabs(["📊 Overview", "📁 Datasets", "▶️ Run Evaluation", "🔍 Results", "⚖️ Compare Runs", "📄 Reports"])

# ================= Overview =================
with tabs[0]:
    runs = db.list_runs()
    datasets = db.list_datasets()
    c1, c2, c3 = st.columns(3)
    c1.metric("Datasets", len(datasets))
    c2.metric("Evaluation runs", len(runs))
    total_traces = 0
    if runs:
        for r in runs:
            total_traces += len(db.get_run_results(r["id"]))
    c3.metric("Traces evaluated", total_traces)

    if runs:
        st.subheader("Recent runs")
        rows = []
        for r in runs[:10]:
            results = db.get_run_results(r["id"])
            scores = [x["overall_score"] for x in results if x["overall_score"] is not None]
            avg_score = round(sum(scores) / len(scores), 3) if scores else None
            avg_cost = round(sum(x["cost_usd"] or 0 for x in results), 4)
            avg_latency = round(sum(x["latency_ms"] or 0 for x in results) / len(results), 1) if results else 0
            rows.append({
                "Run": r["agent_name"], "Model": r["model_name"], "Dataset": r["dataset_name"],
                "Items": len(results), "Avg Score": avg_score, "Total Cost ($)": avg_cost,
                "Avg Latency (ms)": avg_latency, "Created": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No runs yet — go to **Run Evaluation** to evaluate your first agent.")

# ================= Datasets =================
with tabs[1]:
    st.subheader("Evaluation datasets")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Load the sample dataset**")
        if st.button("Load sample dataset (Support Agent Benchmark v1)"):
            with open("data/sample_dataset.json") as f:
                sample = json.load(f)
            did = db.create_dataset(sample["name"])
            for item in sample["items"]:
                db.add_dataset_item(
                    did, item["question"], item["expected_answer"], item["context"],
                    item["relevant_doc_ids"], item["expected_tool_calls"], item.get("tags", []),
                )
            st.success(f"Loaded '{sample['name']}' with {len(sample['items'])} items.")
            st.rerun()

    with right:
        st.markdown("**Upload a custom dataset (JSON)**")
        st.caption("Format: `{\"name\": str, \"items\": [{\"question\", \"expected_answer\", \"context\": [], "
                   "\"relevant_doc_ids\": [], \"expected_tool_calls\": []}]}`")
        uploaded = st.file_uploader("Upload JSON dataset", type=["json"])
        if uploaded is not None and st.button("Import uploaded dataset"):
            data = json.load(uploaded)
            did = db.create_dataset(data["name"])
            for item in data["items"]:
                db.add_dataset_item(
                    did, item["question"], item.get("expected_answer", ""), item.get("context", []),
                    item.get("relevant_doc_ids", []), item.get("expected_tool_calls", []), item.get("tags", []),
                )
            st.success(f"Imported '{data['name']}' with {len(data['items'])} items.")
            st.rerun()

    st.markdown("---")
    datasets = db.list_datasets()
    if datasets:
        chosen = st.selectbox("View dataset", datasets, format_func=lambda d: d["name"], key="view_ds")
        items = db.get_dataset_items(chosen["id"])
        st.dataframe(pd.DataFrame([{
            "Question": i["question"], "Expected Answer": i["expected_answer"],
            "# Context docs": len(i["context"]), "Relevant doc ids": i["relevant_doc_ids"],
            "Expected tool calls": i["expected_tool_calls"], "Tags": i["tags"],
        } for i in items]), use_container_width=True, hide_index=True)
    else:
        st.info("No datasets yet. Load the sample dataset or upload your own above.")

# ================= Run Evaluation =================
with tabs[2]:
    st.subheader("Run an evaluation")
    datasets = db.list_datasets()
    if not datasets:
        st.warning("Create or load a dataset first (see the **Datasets** tab).")
    else:
        dataset = st.selectbox("Dataset", datasets, format_func=lambda d: d["name"], key="run_ds")
        agent_mode = st.radio(
            "Agent under test",
            ["Mock agent (demo, no API needed)", "HTTP endpoint (bring your own agent)", "Anthropic model (built-in minimal agent)"],
            horizontal=False,
        )

        agent_name = st.text_input("Agent / config name for this run", value="agent-v1")
        model_name_field = st.text_input("Model name (for cost tracking, e.g. claude-sonnet-5)", value=judge_model)
        notes = st.text_area("Notes (optional)", placeholder="e.g. testing new system prompt with tighter tool-use instructions")

        endpoint_url, headers_raw = None, ""
        if agent_mode.startswith("HTTP"):
            endpoint_url = st.text_input("Agent endpoint URL",
                                          placeholder="https://your-agent.example.com/invoke")
            headers_raw = st.text_input("Extra headers (JSON, optional)", placeholder='{"Authorization": "Bearer ..."}')

        run_clicked = st.button("▶️ Run evaluation", type="primary")

        if run_clicked:
            items = db.get_dataset_items(dataset["id"])
            if not items:
                st.error("This dataset has no items.")
            else:
                if agent_mode.startswith("HTTP") and not endpoint_url:
                    st.error("Please provide an agent endpoint URL.")
                elif agent_mode.startswith("Anthropic") and not judge_client:
                    st.error("Please provide an Anthropic API key in the sidebar to use this mode.")
                else:
                    run_id = db.create_run(dataset["id"], agent_name, model_name_field, notes)
                    progress = st.progress(0.0, text="Running...")
                    headers = {}
                    if headers_raw.strip():
                        try:
                            headers = json.loads(headers_raw)
                        except Exception:
                            st.warning("Could not parse headers JSON; ignoring.")

                    for idx, item in enumerate(items):
                        try:
                            if agent_mode.startswith("Mock"):
                                trace = run_mock_agent(item)
                            elif agent_mode.startswith("HTTP"):
                                trace = run_http_agent(item, endpoint_url, headers)
                            else:
                                trace = run_anthropic_agent(item, judge_client, judge_model)
                        except Exception as e:
                            trace = {"answer": f"[ERROR calling agent: {e}]", "retrieved_doc_ids": [],
                                      "tool_calls": [], "latency_ms": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "raw_trace": {"error": str(e)}}

                        cost = estimate_cost(model_name_field, trace["prompt_tokens"], trace["completion_tokens"])
                        trace_id = db.add_trace(
                            run_id, item["id"], item["question"], trace["answer"],
                            trace["retrieved_doc_ids"], trace["tool_calls"], trace["latency_ms"],
                            trace["prompt_tokens"], trace["completion_tokens"], cost, trace.get("raw_trace"),
                        )
                        evald = evaluate_trace(item, trace, use_llm_judge=use_llm_judge,
                                                judge_client=judge_client, judge_model=judge_model)
                        db.add_evaluation(
                            trace_id, run_id, evald["correctness"], evald["faithfulness"],
                            evald["rag_precision"], evald["rag_recall"], evald["tool_call_accuracy"],
                            evald["safety_flag"], evald["safety_reason"], evald["overall_score"], evald["details"],
                        )
                        progress.progress((idx + 1) / len(items), text=f"Evaluated {idx + 1}/{len(items)}")

                    st.success(f"Run complete! {len(items)} items evaluated. See the **Results** tab.")
                    st.session_state["last_run_id"] = run_id

# ================= Results =================
with tabs[3]:
    st.subheader("Run results")
    runs = db.list_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        default_idx = 0
        if "last_run_id" in st.session_state:
            ids = [r["id"] for r in runs]
            if st.session_state["last_run_id"] in ids:
                default_idx = ids.index(st.session_state["last_run_id"])
        chosen_run = st.selectbox(
            "Select run", runs,
            format_func=lambda r: f"{r['agent_name']} ({r['model_name']}) — {r['dataset_name']} — "
                                   f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created_at']))}",
            index=default_idx,
        )
        results = db.get_run_results(chosen_run["id"])
        df = pd.DataFrame(results)

        if not df.empty:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Avg correctness", f"{df['correctness'].dropna().mean():.2f}" if df["correctness"].notna().any() else "—")
            m2.metric("Avg faithfulness", f"{df['faithfulness'].dropna().mean():.2f}" if df["faithfulness"].notna().any() else "—")
            m3.metric("Avg overall score", f"{df['overall_score'].dropna().mean():.2f}" if df["overall_score"].notna().any() else "—")
            m4.metric("Total cost", f"${df['cost_usd'].sum():.4f}")
            m5.metric("Avg latency", f"{df['latency_ms'].mean():.0f} ms")

            flagged = df[df["safety_flag"] == 1]
            if len(flagged):
                st.error(f"⚠️ {len(flagged)} response(s) flagged by the safety checker.")

            st.markdown("#### Per-item results")
            display_df = df[["question", "answer", "correctness", "faithfulness", "rag_precision",
                              "rag_recall", "tool_call_accuracy", "overall_score", "latency_ms", "cost_usd", "safety_flag"]].copy()
            display_df.columns = ["Question", "Answer", "Correctness", "Faithfulness", "RAG Precision",
                                   "RAG Recall", "Tool Accuracy", "Overall", "Latency (ms)", "Cost ($)", "Flagged"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            with st.expander("🔬 Inspect a single trace"):
                q_options = df["question"].tolist()
                sel_q = st.selectbox("Question", q_options)
                row = df[df["question"] == sel_q].iloc[0]
                st.write("**Answer:**", row["answer"])
                st.json(row["details"])

            if st.button("🗑️ Delete this run"):
                db.delete_run(chosen_run["id"])
                st.rerun()
        else:
            st.info("This run has no traces.")

# ================= Compare Runs =================
with tabs[4]:
    st.subheader("Compare two runs (regression detection)")
    runs = db.list_runs()
    if len(runs) < 2:
        st.info("Need at least 2 runs to compare. Run evaluations on different agent versions/models first.")
    else:
        fmt = lambda r: f"{r['agent_name']} ({r['model_name']}) — {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created_at']))}"
        c1, c2 = st.columns(2)
        run_a = c1.selectbox("Baseline run", runs, format_func=fmt, key="cmp_a")
        run_b = c2.selectbox("Candidate run", runs, format_func=fmt, index=1, key="cmp_b")

        ra, rb = db.get_run_results(run_a["id"]), db.get_run_results(run_b["id"])
        dfa, dfb = pd.DataFrame(ra), pd.DataFrame(rb)

        if dfa.empty or dfb.empty:
            st.warning("One of the selected runs has no data.")
        else:
            metrics = ["correctness", "faithfulness", "rag_precision", "rag_recall", "tool_call_accuracy", "overall_score"]
            rows = []
            for m in metrics:
                va = dfa[m].dropna().mean() if dfa[m].notna().any() else None
                vb = dfb[m].dropna().mean() if dfb[m].notna().any() else None
                delta = round(vb - va, 3) if va is not None and vb is not None else None
                rows.append({"Metric": m, "Baseline": round(va, 3) if va is not None else "—",
                             "Candidate": round(vb, 3) if vb is not None else "—",
                             "Δ": delta, "Regression?": "⚠️ Yes" if (delta is not None and delta < -0.03) else ("✅ Improved" if (delta is not None and delta > 0.03) else "—")})
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            cost_a, cost_b = dfa["cost_usd"].sum(), dfb["cost_usd"].sum()
            lat_a, lat_b = dfa["latency_ms"].mean(), dfb["latency_ms"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Total cost", f"${cost_b:.4f}", delta=f"{cost_b - cost_a:+.4f}", delta_color="inverse")
            c2.metric("Avg latency", f"{lat_b:.0f} ms", delta=f"{lat_b - lat_a:+.0f} ms", delta_color="inverse")

            chart_df = cmp_df[cmp_df["Baseline"] != "—"].melt(id_vars="Metric", value_vars=["Baseline", "Candidate"],
                                                                var_name="Run", value_name="Score")
            chart_df["Score"] = pd.to_numeric(chart_df["Score"], errors="coerce")
            fig = px.bar(chart_df, x="Metric", y="Score", color="Run", barmode="group",
                         title="Baseline vs Candidate — quality metrics")
            st.plotly_chart(fig, use_container_width=True)

# ================= Reports =================
with tabs[5]:
    st.subheader("Export a report")
    runs = db.list_runs()
    if not runs:
        st.info("No runs to report on yet.")
    else:
        fmt = lambda r: f"{r['agent_name']} ({r['model_name']}) — {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created_at']))}"
        chosen = st.selectbox("Run", runs, format_func=fmt, key="report_run")
        results = db.get_run_results(chosen["id"])
        df = pd.DataFrame(results)

        if not df.empty:
            lines = [
                f"# Evaluation Report — {chosen['agent_name']} ({chosen['model_name']})",
                f"Dataset: {chosen['dataset_name']}",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
                "",
                "## Summary",
                f"- Items evaluated: {len(df)}",
                f"- Avg correctness: {df['correctness'].dropna().mean():.3f}" if df["correctness"].notna().any() else "- Avg correctness: N/A",
                f"- Avg faithfulness: {df['faithfulness'].dropna().mean():.3f}" if df["faithfulness"].notna().any() else "- Avg faithfulness: N/A",
                f"- Avg overall score: {df['overall_score'].dropna().mean():.3f}" if df["overall_score"].notna().any() else "- Avg overall score: N/A",
                f"- Total cost: ${df['cost_usd'].sum():.4f}",
                f"- Avg latency: {df['latency_ms'].mean():.0f} ms",
                f"- Safety flags: {int((df['safety_flag'] == 1).sum())}",
                "",
                "## Per-item detail",
            ]
            for _, row in df.iterrows():
                lines.append(f"### Q: {row['question']}")
                lines.append(f"- Answer: {row['answer']}")
                lines.append(f"- Overall score: {row['overall_score']}")
                if row["safety_flag"]:
                    lines.append(f"- ⚠️ Safety flag: {row['safety_reason']}")
                lines.append("")
            report_md = "\n".join(lines)
            st.download_button("⬇️ Download report (Markdown)", report_md,
                                file_name=f"eval_report_{chosen['agent_name']}.md", mime="text/markdown")
            st.markdown("---")
            st.markdown(report_md)
