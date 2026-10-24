"""Turn measured result files into LaTeX macros and charts for the paper.

Every number in the paper's results is a macro written here from a JSON file
produced by an actual run, so the PDF cannot drift from the data.

    python scripts/make_paper_artifacts.py
"""
from __future__ import annotations

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES_GPU = ROOT / "results_gpu"
OUT_TEX = ROOT / "paper" / "results_macros.tex"
FIG = ROOT / "paper" / "figures"
CLASSES = ["account_number", "date", "amount", "name", "signature"]
LABEL = {"account_number": "account no.", "date": "date", "amount": "amount", "name": "name", "signature": "signature"}
VARIANTS = {"Def": "default", "Free": "nms_free", "Nms": "class_nms"}


def load(name: str, base: Path | None = None) -> dict:
    return json.loads(((base or RES) / name).read_text())


def pct(x: float, nd: int = 1) -> str:
    """Percent string, rounding halves up (91.25 -> 91.3) rather than to even."""
    q = Decimal(1).scaleb(-nd)
    return str((Decimal(str(round(x, 6))) * 100).quantize(q, rounding=ROUND_HALF_UP))


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    train, pat = load("train_summary.json"), load("pattern.json")
    ev = {k: load(f"eval_report_{v}.json") for k, v in VARIANTS.items()}
    run = {k: load(f"run_report_{v}.json") for k, v in VARIANTS.items()}
    m: dict[str, str] = {}
    N_TEXT, N_FIELDS = 240, 300

    m["MapFifty"] = f"{train['val_map50']:.3f}"
    m["MapFiftyNinetyFive"] = f"{train['val_map50_95']:.3f}"
    m["TrainMinutes"] = f"{train['train_seconds'] / 60:.0f}"
    m["TrainEpochs"] = str(train["epochs"])

    for k in VARIANTS:
        e, r = ev[k], run[k]
        m[f"{k}Prec"], m[f"{k}Rec"] = pct(e["overall_precision"]), pct(e["overall_recall"])
        m[f"{k}EndToEnd"] = pct(e["pipeline_end_to_end_field_accuracy"])
        m[f"{k}Conf"] = f"{r['conf']:.2f}"
        m[f"{k}Fields"], m[f"{k}Seconds"] = str(r["fields"]), f"{r['seconds']:.0f}"
        lo, hi = wilson(round(e["pipeline_end_to_end_field_accuracy"] * N_TEXT), N_TEXT)
        m[f"{k}EndToEndLo"], m[f"{k}EndToEndHi"] = pct(lo), pct(hi)
        # OCR stage only: exact match among correctly localized text fields
        tot_ok = sum(round(c["ocr_exact"] * round(c["recall"] * c["n"])) for c in e["per_class"].values() if c["ocr_exact"] is not None)
        tot_n = sum(round(c["recall"] * c["n"]) for c in e["per_class"].values() if c["ocr_exact"] is not None)
        m[f"{k}OcrStage"] = pct(tot_ok / tot_n) if tot_n else "--"
    m["NmsExtra"] = str(run["Nms"]["fields"] - round(ev["Nms"]["overall_recall"] * N_FIELDS))
    m["NmsChosenValF"] = pct(ev["Nms"]["val_f1_at_chosen"])
    m["FreeChosenValF"] = pct(ev["Free"]["val_f1_at_chosen"])
    m["BaseCapture"] = pct(ev["Nms"]["fullpage_tesseract_field_capture"])
    lo, hi = wilson(round(ev["Nms"]["fullpage_tesseract_field_capture"] * N_TEXT), N_TEXT)
    m["BaseLo"], m["BaseHi"] = pct(lo), pct(hi)
    m["BaseSecPerDoc"] = f"{ev['Nms']['fullpage_tesseract_seconds_per_doc']:.2f}"
    d = ev["Def"]
    m["DefDup"], m["DefMissed"] = str(d["duplicate_boxes"]), str(d["missed_fields"])
    m["DefDupConf"], m["DefTopConf"] = f"{d['duplicate_mean_conf']:.2f}", f"{d['top_box_mean_conf']:.2f}"
    errs = ev["Nms"]["ocr_errors"]
    m["NmsOcrErrCount"] = str(len(errs))
    m["NmsOcrErrAllAccount"] = "all" if errs and all(e["cls"] == "account_number" for e in errs) else "not all"
    def q(x): return "\\texttt{" + x + "}" if x else "an empty string"
    m["OcrErrExamples"] = "; ".join(f"{q(e['truth'])} read as {q(e['read'])}" for e in errs[:3]) if errs else "none"
    for c in CLASSES:
        pc, key = ev["Nms"]["per_class"][c], "".join(w.capitalize() for w in c.split("_"))
        m[f"Prec{key}"], m[f"Rec{key}"] = pct(pc["precision"]), pct(pc["recall"])
        m[f"Ocr{key}"] = pct(pc["ocr_exact"]) if pc["ocr_exact"] is not None else "n/a"

    modes = pat["modes"]
    m["PatDocs"], m["PatParts"], m["PatRepeats"] = str(pat["docs"]), str(pat["partitions"]), str(pat["repeats"])
    for key, short in [("row_udf", "RowUdf"), ("partition_b1", "PartOne"), ("partition_b16", "PartSixteen")]:
        m[f"Pat{short}Sec"], m[f"Pat{short}Sd"] = f"{modes[key]['seconds_mean']:.1f}", f"{modes[key]['seconds_sd']:.1f}"
        m[f"Pat{short}Dps"], m[f"Pat{short}Speedup"] = f"{modes[key]['docs_per_second']:.1f}", f"{modes[key]['speedup_vs_row_udf']:.1f}"
    m["PatBoxes"] = str(modes["row_udf"]["total_boxes"][0])
    m["PatLoadMs"] = f"{1000 * (modes['row_udf']['seconds_mean'] - modes['partition_b16']['seconds_mean']) / (pat['docs'] / pat['partitions'] - 1):.0f}"

    # GPU replication (Colab T4). Same test images (hash checked below), separate training run.
    if (RES_GPU / "environment.json").exists():
        env = load("environment.json", RES_GPU)
        gt, gp = load("train_summary.json", RES_GPU), load("pattern.json", RES_GPU)
        m["GpuName"] = env["gpu"].split(",")[0].replace("Tesla ", "")
        m["GpuTestHash"] = env["test_images_sha256"][:8]
        cpu_env = load("environment.json")
        m["GpuHashMatches"] = "identical" if env["test_images_sha256"] == cpu_env["test_images_sha256"] else "different"
        m["GpuProvider"] = env["ultralytics_session_providers"][0].replace("ExecutionProvider", "")
        m["GpuTrainSeconds"] = f"{gt['train_seconds']:.0f}"
        m["GpuMapFifty"], m["GpuMapFiftyNinetyFive"] = f"{gt['val_map50']:.3f}", f"{gt['val_map50_95']:.3f}"
        for k, v in VARIANTS.items():
            e, r = load(f"eval_report_{v}.json", RES_GPU), load(f"run_report_{v}.json", RES_GPU)
            m[f"Gpu{k}Prec"], m[f"Gpu{k}Rec"] = pct(e["overall_precision"]), pct(e["overall_recall"])
            m[f"Gpu{k}EndToEnd"], m[f"Gpu{k}Conf"] = pct(e["pipeline_end_to_end_field_accuracy"]), f"{r['conf']:.2f}"
            m[f"Gpu{k}Dup"], m[f"Gpu{k}Missed"] = str(e["duplicate_boxes"]), str(e["missed_fields"])
            m[f"Gpu{k}Extra"] = str(r["fields"] - round(e["overall_recall"] * N_FIELDS))
            lo, hi = wilson(round(e["pipeline_end_to_end_field_accuracy"] * N_TEXT), N_TEXT)
            m[f"Gpu{k}EndToEndLo"], m[f"Gpu{k}EndToEndHi"] = pct(lo), pct(hi)
        fb = ROOT / "results_gpu_font_fallback"
        if (fb / "eval_report_class_nms.json").exists():
            fe = load("eval_report_class_nms.json", fb)
            m["FallbackEndToEnd"] = pct(fe["pipeline_end_to_end_field_accuracy"])
            m["FallbackNmsRec"] = pct(fe["overall_recall"])
            assert not load("environment.json", fb)["fonts_found"], "fallback run must have had no fonts"
        gm = gp["modes"]
        for key, short in [("row_udf", "RowUdf"), ("partition_b1", "PartOne"), ("partition_b16", "PartSixteen")]:
            m[f"GpuPat{short}Sec"], m[f"GpuPat{short}Sd"] = f"{gm[key]['seconds_mean']:.1f}", f"{gm[key]['seconds_sd']:.1f}"
            m[f"GpuPat{short}Speedup"] = f"{gm[key]['speedup_vs_row_udf']:.1f}"
        m["GpuPatBoxes"] = str(gm["row_udf"]["total_boxes"][0])
        m["GpuPatBoxesSame"] = "the same" if len({tuple(x["total_boxes"]) for x in gm.values()}) == 1 else "different"

    OUT_TEX.write_text("% AUTO-GENERATED by scripts/make_paper_artifacts.py. Do not edit.\n" +
                       "".join(f"\\newcommand{{\\val{k}}}{{{v}}}\n" for k, v in m.items()))

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    xs, w = range(len(CLASSES)), 0.26
    e = ev["Nms"]["per_class"]
    ax.bar([x - w for x in xs], [100 * e[c]["precision"] for c in CLASSES], w, label="detection precision", color="#1F4E8C")
    ax.bar(list(xs), [100 * e[c]["recall"] for c in CLASSES], w, label="detection recall", color="#6FA8DC")
    ax.bar([x + w for x in xs], [100 * (e[c]["ocr_exact"] or 0) for c in CLASSES], w, label="OCR exact match", color="#B26A00")
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL[c] for c in CLASSES], rotation=20)
    ax.set_ylabel("percent"); ax.set_ylim(60, 104); ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "per_class.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    labs = ["NMS-free\nconf 0.25", "NMS-free\ntuned", "class NMS\ntuned"]
    keys = ["Def", "Free", "Nms"]
    xs, w = range(3), 0.26
    for off, (name, fld, col) in zip((-w, 0, w), [("precision", "overall_precision", "#1F4E8C"), ("recall", "overall_recall", "#6FA8DC"),
                                                  ("end-to-end", "pipeline_end_to_end_field_accuracy", "#B26A00")]):
        ax.bar([x + off for x in xs], [100 * ev[k][fld] for k in keys], w, label=name, color=col)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labs); ax.set_ylabel("percent"); ax.set_ylim(60, 104)
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "variants.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    order = [("row_udf", "row-wise UDF\n(model per row)"), ("partition_b1", "per-partition\nbatch 1"), ("partition_b16", "per-partition\nbatch 16")]
    means = [modes[k]["seconds_mean"] for k, _ in order]
    bars = ax.bar([l for _, l in order], means, yerr=[modes[k]["seconds_sd"] for k, _ in order], capsize=3,
                  color=["#B00020", "#6FA8DC", "#1B7F3B"], width=0.55)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.45, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_ylabel(f"seconds for {pat['docs']} docs"); ax.set_ylim(0, max(means) * 1.25)
    fig.tight_layout(); fig.savefig(FIG / "pattern.png", dpi=300); plt.close(fig)
    print("wrote", OUT_TEX, "and 3 figures")


if __name__ == "__main__":
    main()
