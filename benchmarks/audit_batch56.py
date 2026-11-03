"""Audit retained batch evidence against actual sources, artifacts and check logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("node_count", "byte_size", "output_count", "input_count", "initializer_count")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost(record):
    return tuple(record["report"]["final"][key] for key in FIELDS)


def audit_artifact(record):
    report = record["report"]
    path = Path(report["artifact_recheck"]["model"])
    assert digest(path) == report["final_hash"] == report["artifact_recheck"]["digest"]
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    expected = (
        len(model.graph.node),
        path.stat().st_size,
        len(model.graph.output),
        len(model.graph.input),
        len(model.graph.initializer),
    )
    assert cost(record) == expected, path
    checks = [
        json.loads(line)
        for line in (Path(report["work_directory"]) / "evaluations.jsonl").read_text().splitlines()
    ]
    assert not checks[0]["cache_hit"] and not checks[-1]["cache_hit"]
    assert checks[-1]["model"] == str(path)
    assert checks[-1]["outcome"] == report["artifact_recheck"]["outcome"] == "interesting"
    assert report["artifact_recheck"]["read_back_validated"]
    if report["minimality"]["status"] == "pass_fixed_point":
        assert report["scheduler"]["coverage"]["completed"]
        assert report["scheduler"]["coverage"]["base_digest"] == report["final_hash"]
        assert report["scheduler"]["coverage"]["inconclusive_candidate_count"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", default=[])
    parser.add_argument("--extended", type=Path, action="append", default=[])
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.batch = [p.resolve() for p in args.batch]
    args.extended = [p.resolve() for p in args.extended]
    args.cli = args.cli.resolve()
    total = 0
    batches = []
    compatibility = 0
    search_comparisons = {"equal": 0, "improved": 0, "regressed": 0}
    current = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted([*ROOT.glob("src/onnx_reduce/**/*.py"), ROOT / "benchmarks/reduction.py"])
    }
    for manifest in args.batch:
        data = json.loads(manifest.read_text())
        assert data["source_sha256"]["current"] == current, "measured sources changed"
        rows = [
            json.loads(line) for line in (manifest.parent / "runs.jsonl").read_text().splitlines()
        ]
        assert len(rows) == data["count"]
        for row in rows:
            audit_artifact(row)
            if row["budget"] == "calls":
                command = row["command"]
                quota = int(
                    [
                        command[i + 1]
                        for i, value in enumerate(command)
                        if value == "--candidate-budget"
                    ][-1]
                )
                assert row["report"]["failure_checks"]["calls"] <= quota + 2
            if data["suite"] == "preparation" and row["variant"] != "before":
                control = next(
                    r
                    for r in rows
                    if r["scenario"] == row["scenario"]
                    and r["variant"] == "before"
                    and r["repeat"] == row["repeat"]
                )
                for key in ("decision_trace_sha256", "scheduler_trace_sha256"):
                    assert control[key] == row[key], (row["scenario"], key)
                compatibility += 1
            if (
                data["suite"] == "search"
                and row["budget"] == "calls"
                and row["variant"]
                in {
                    "cursor-complete",
                    "adaptive",
                    "history",
                }
            ):
                controls = [
                    r
                    for r in rows
                    if r["scenario"] == row["scenario"]
                    and r["variant"] == "restart-complete"
                    and r["repeat"] == row["repeat"]
                ]
                if controls:
                    before, after = cost(controls[0]), cost(row)
                    search_comparisons[
                        "improved" if after < before else "regressed" if after > before else "equal"
                    ] += 1
        total += len(rows)
        batches.append(
            {
                "manifest": str(manifest.relative_to(ROOT)),
                "sha256": digest(manifest),
                "count": len(rows),
                "driver_sha256": data["driver_sha256"],
                "source_sha256": data["source_sha256"],
                "command": data["command"],
                "environment": data["environment"],
                "summaries": data["summaries"],
            }
        )
    extended = []
    extended_rows = []
    for manifest in args.extended:
        data = json.loads(manifest.read_text())
        expected_source = (
            batches[0]["source_sha256"]["before"]
            if "before" in data["configuration"]["output"]
            else current
        )
        for name, sha in data["environment"]["source_sha256"].items():
            if name.startswith("src/"):
                assert expected_source[name] == sha
        for row in data["runs"]:
            audit_artifact(row)
            extended_rows.append((data, row))
        total += len(data["runs"])
        extended.append(
            {
                "manifest": str(manifest.relative_to(ROOT)),
                "sha256": digest(manifest),
                "count": len(data["runs"]),
                "configuration": data["configuration"],
                "environment": data["environment"],
                "summaries": data["summaries"],
            }
        )
    for data, row in extended_rows:
        if row["budget"] == "calls" and data["configuration"].get("model_edit_mode") == "auto":
            controls = [
                other
                for config, other in extended_rows
                if "before" in config["configuration"]["output"]
                and other["case"] == row["case"]
                and other["scheduler"] == row["scheduler"]
                and other["repeat"] == row["repeat"]
            ]
            assert len(controls) == 1
            control = controls[0]
            assert cost(control) == cost(row)
            assert control["report"]["final_hash"] == row["report"]["final_hash"]
            for key in ("decision_trace_sha256", "scheduler_trace_sha256"):
                assert control[key] == row[key]
            compatibility += 1
    cli = {}
    for name in ("fast", "final"):
        path = args.cli / f"{name}.json"
        report = json.loads(path.read_text())
        audit_artifact({"report": report})
        assert report["minimality"]["status"] == (
            "unproven" if name == "fast" else "pass_fixed_point"
        )
        cli[name] = {
            "report": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "final_hash": report["final_hash"],
            "optimizations": report["optimizations"],
            "minimality": report["minimality"],
        }
    output = {
        "version": 1,
        "baseline_commit": "3f7cb3a64d84c6ce1324a94da24c9f3cd2c86fd1",
        "reference_commit": "666fc9b545596c61893d5afc882e3e9bae7299f7",
        "audited_benchmark_runs": total,
        "compatibility_pairs": compatibility,
        "search_cost_comparisons": search_comparisons,
        "current_source_sha256": current,
        "batches": batches,
        "extended": extended,
        "cli": cli,
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Audited {total} benchmark artifacts; {compatibility} exact compatibility pairs; {search_comparisons}"
    )


if __name__ == "__main__":
    main()
