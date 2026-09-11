from __future__ import annotations

"""Tests for the Phase-2.75D hard-landscape dynamics diagnosis.

The pure trace statistics are checked against hand-computed numbers, the
end-to-end diagnosis runs on a two-snapshot pop-20 zdt1 fixture, and the
``--report`` mode is exercised on crafted diagnosis payloads so each of the
four candidate causes is driven to its documented verdict.

No real diagnosis is run: the fixture branches 2 states x 5 candidates x 2
replicates x 4 generations of pop 20.
"""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from controller.multihead_controller import MultiHeadController
from controller.state_encoder import StateEncoder
from experiments import diagnose_hard_landscape as dhl

_POP = 20
_GENS = 6
_STATES = 2
_BRANCH_GENERATIONS = 4
_N_ALTERNATIVES = 3
_N_REPS = 2


# --- pure statistics ---------------------------------------------------------


def test_stagnation_and_escape_hand_computed() -> None:
    """Longest stagnation run and first escape generations, hand-checked."""
    # gains: [0.0, 0.05, 0.0, 0.0, 0.15]; eps = 0.01
    trace = [0.0, 0.0, 0.05, 0.05, 0.05, 0.20]
    result = dhl.stagnation_and_escape(
        trace, epsilon=0.01, hv_reference=1.0, thresholds=(0.10, 0.50)
    )
    assert result["stagnation_gens"] == 2  # indices 2 and 3
    assert result["stagnation_gens_total"] == 3  # indices 0, 2, 3
    assert result["escape_gen_10pct"] == 5  # first hv >= 0.10
    assert result["escape_gen_50pct"] is None  # never reaches 0.50
    assert result["escape_thresholds"] == {
        "escape_gen_10pct": pytest.approx(0.10),
        "escape_gen_50pct": pytest.approx(0.50),
    }
    assert result["final_hv"] == pytest.approx(0.20)
    assert result["hv_gain_total"] == pytest.approx(0.20)
    # an already-escaped snapshot reports generation 0
    early = dhl.stagnation_and_escape(
        [0.6, 0.6, 0.7], epsilon=0.01, hv_reference=1.0
    )
    assert early["escape_gen_10pct"] == 0
    assert early["escape_gen_50pct"] == 0
    assert early["stagnation_gens"] == 1
    # no reference hypervolume -> no escape information, but stagnation remains
    without_reference = dhl.stagnation_and_escape([0.1, 0.1, 0.1], epsilon=0.01)
    assert without_reference["escape_gen_10pct"] is None
    assert without_reference["stagnation_gens"] == 2
    assert without_reference["stagnation_gens_total"] == 2
    with pytest.raises(ValueError, match="must not be empty"):
        dhl.stagnation_and_escape([], epsilon=0.01)
    # a single-entry trace has no gains to stagnate on
    single = dhl.stagnation_and_escape([0.5], epsilon=0.01, hv_reference=1.0)
    assert single["stagnation_gens"] == 0
    assert single["stagnation_gens_total"] == 0
    assert single["escape_gen_50pct"] == 0


def test_exploration_duration_hand_computed() -> None:
    """Duration counts generations at or above the fraction of gen-1 spread."""
    trace = [0.4, 0.4, 0.3, 0.2, 0.1]
    assert dhl.exploration_duration(trace, fraction=0.5) == 4  # >= 0.2
    assert dhl.exploration_duration(trace, fraction=0.8) == 2  # >= 0.32
    assert dhl.exploration_duration(trace, fraction=1.0) == 2  # >= 0.4
    assert dhl.exploration_duration([0.1], fraction=0.5) == 0
    assert dhl.exploration_duration([0.1, 0.0], fraction=0.5) == 0
    with pytest.raises(ValueError, match="fraction"):
        dhl.exploration_duration(trace, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        dhl.exploration_duration(trace, fraction=1.5)


def test_action_effect_snr_hand_computed() -> None:
    """Between-action variance over replicate noise."""
    rows = [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]]
    between = float(np.var([0.05, 1.05, 2.05], ddof=1))
    within = float(np.mean([np.var(row, ddof=1) for row in rows]))
    assert dhl.action_effect_snr(rows) == pytest.approx(between / within)
    assert dhl.action_effect_snr([[1.0, 1.0], [2.0, 2.0]]) is None  # zero noise
    assert dhl.action_effect_snr([[1.0, 1.0]]) is None  # one candidate
    assert dhl.action_effect_snr([[1.0], [2.0]]) is None  # one replicate


def test_mean_median_helpers() -> None:
    """Replicate aggregation helpers skip undefined entries."""
    assert dhl._mean_trace([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]
    assert dhl._mean_trace([]) == []
    assert dhl._median_or_none([1.0, None, 3.0]) == pytest.approx(2.0)
    assert dhl._median_or_none([None, None]) is None
    assert dhl._fraction_defined([1, None, 2, 3]) == pytest.approx(0.75)
    assert dhl._fraction_defined([]) is None


# --- end-to-end diagnosis ----------------------------------------------------


@pytest.fixture(scope="module")
def diagnosis(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A tiny zdt1 diagnosis (2 states x 5 candidates x 2 reps x 4 generations)."""
    root = tmp_path_factory.mktemp("hard_landscape")
    trajectories = []
    for j in range(2):
        hv, igd = 0.30 + 0.05 * j, 0.50 - 0.02 * j
        transitions = []
        for t in range(_GENS + 1):
            delta_hv = 0.0 if t == 0 else 0.006 + 0.001 * ((t + j) % 3)
            delta_igd = 0.0 if t == 0 else 0.005 + 0.001 * ((t + j + 1) % 3)
            hv += delta_hv
            igd -= delta_igd
            transitions.append(
                {
                    "generation": t,
                    "state": {"generation": t, "hv": hv, "igd": igd,
                              "diversity": 0.25 + 0.01 * t},
                    "action": {"mutation_operator": "polynomial",
                               "mutation_probability": 1.0 / 30.0,
                               "exploration_strength": 20.0},
                    "reward": {"delta_hv": delta_hv, "delta_igd": delta_igd},
                }
            )
        trajectories.append(transitions)
    encoder = StateEncoder(3).fit(trajectories)
    encoder_path = root / "encoder.json"
    encoder.save(encoder_path)
    controller_path = root / "controller.pt"
    MultiHeadController(input_dim=encoder.dim, seed=0, name="hl_test").save(
        controller_path
    )
    from experiments.counterfactual_actions import harvest_snapshots

    harvest_snapshots(
        "zdt1",
        seed=1000,
        generations=_GENS,
        pop_size=_POP,
        states_per_run=_STATES,
        n_reference_points=200,
        ref_point=np.asarray([1.1, 1.1]),
        out_dir=root / "snapshots",
    )
    out_path = root / "diagnosis.json"
    payload = dhl.main(
        [
            "--problem", "zdt1",
            "--snapshots-dir", str(root / "snapshots"),
            "--max-states", str(_STATES),
            "--generations", str(_BRANCH_GENERATIONS),
            "--n-alternatives", str(_N_ALTERNATIVES),
            "--n-reps", str(_N_REPS),
            "--controller", str(controller_path),
            "--controller-type", "multihead",
            "--encoder", str(encoder_path),
            "--out", str(out_path),
        ]
    )
    return {"root": root, "out": out_path, "payload": payload}


def test_diagnosis_fields_and_trace_lengths(diagnosis: dict[str, Any]) -> None:
    """Every documented per-candidate field is present and well sized."""
    payload = diagnosis["payload"]
    assert payload["problem"] == "zdt1"
    config = payload["config"]
    assert config["generations"] == _BRANCH_GENERATIONS
    assert config["trace_length"] == _BRANCH_GENERATIONS + 1
    assert config["n_states_evaluated"] == _STATES
    assert config["include_default_action"] is True
    # 2 states x (1 controller + 1 default + 3 alternatives)
    assert len(payload["per_candidate"]) == _STATES * (2 + _N_ALTERNATIVES)
    required = {
        "index", "kind", "action", "hv_trace", "igd_trace", "diversity_trace",
        "population_std_trace", "population_span_trace",
        "population_abs_change_trace", "mutation_probability_trace",
        "mutation_operator_trace", "exploration_strength_trace",
        "final_hv", "final_hv_mean",
        "stagnation_gens", "stagnation_gens_total", "stagnation_epsilon",
        "escape_gen_10pct", "escape_gen_50pct", "exploration_duration",
        "exploration_std_fraction", "n_reps", "generations",
    }
    for entry in payload["per_candidate"]:
        assert required <= set(entry)
        assert entry["kind"] in {"controller", "default", "alternative"}
        for key in ("hv_trace", "igd_trace", "diversity_trace",
                    "population_std_trace", "population_span_trace",
                    "population_abs_change_trace", "mutation_probability_trace",
                    "exploration_strength_trace"):
            assert len(entry[key]) == _BRANCH_GENERATIONS + 1
            assert all(isinstance(value, float) for value in entry[key])
        # the applied action is constant along a branch and recorded every
        # generation, exactly as the plan asks
        assert len(entry["mutation_operator_trace"]) == _BRANCH_GENERATIONS + 1
        assert set(entry["mutation_operator_trace"]) == {
            entry["action"]["mutation_operator"]
        }
        assert entry["mutation_probability_trace"] == pytest.approx(
            [entry["action"]["mutation_probability"]] * (_BRANCH_GENERATIONS + 1)
        )
        assert len(entry["final_hv"]) == _N_REPS
        assert entry["final_hv_mean"] == pytest.approx(np.mean(entry["final_hv"]))
        assert set(entry["action"]) == {
            "mutation_operator", "mutation_probability", "exploration_strength"
        }
        assert entry["exploration_duration"] >= 0.0
        assert entry["stagnation_gens"] >= 0.0
        assert entry["n_reps"] == _N_REPS
    summary = payload["summary"]
    assert summary["n_states"] == _STATES
    assert set(summary["best_action_by_final_hv"]) >= {
        "n_states", "planner_is_best_fraction", "kind_counts", "per_state"
    }
    rank = summary["planner_action_rank_by_final_hv"]
    assert 0.0 <= rank["mean_percentile"] <= 1.0
    assert set(summary["mean_escape_gen_by_action_kind"]) <= {
        "controller", "default", "alternative"
    }
    assert len(summary["per_state"]) == _STATES


def test_diagnosis_is_deterministic(diagnosis: dict[str, Any]) -> None:
    """Re-running the same grid reproduces the traces bit for bit."""
    payload = diagnosis["payload"]
    root = diagnosis["root"]
    again = dhl.main(
        [
            "--problem", "zdt1",
            "--snapshots-dir", str(root / "snapshots"),
            "--max-states", str(_STATES),
            "--generations", str(_BRANCH_GENERATIONS),
            "--n-alternatives", str(_N_ALTERNATIVES),
            "--n-reps", str(_N_REPS),
            "--controller", str(root / "controller.pt"),
            "--controller-type", "multihead",
            "--encoder", str(root / "encoder.json"),
            "--out", str(root / "diagnosis_again.json"),
        ]
    )
    first = payload["per_candidate"][0]["hv_trace"]
    second = again["per_candidate"][0]["hv_trace"]
    assert first == pytest.approx(second)
    assert payload["summary"]["planner_action_rank_by_final_hv"] == (
        again["summary"]["planner_action_rank_by_final_hv"]
    )


def test_per_rep_storage_and_default_toggle(
    diagnosis: dict[str, Any], tmp_path: Path
) -> None:
    """--store-per-rep adds the raw traces; the default candidate can be off."""
    root = diagnosis["root"]
    payload = dhl.main(
        [
            "--problem", "zdt1",
            "--snapshots-dir", str(root / "snapshots"),
            "--max-states", "1",
            "--generations", "2",
            "--n-alternatives", "2",
            "--n-reps", "2",
            "--controller", str(root / "controller.pt"),
            "--controller-type", "multihead",
            "--encoder", str(root / "encoder.json"),
            "--out", str(tmp_path / "per_rep.json"),
            "--store-per-rep",
            "--no-include-default-action",
        ]
    )
    assert payload["config"]["include_default_action"] is False
    assert payload["config"]["store_per_rep"] is True
    entry = payload["per_candidate"][0]
    assert len(entry["per_rep"]) == 2
    assert len(entry["per_rep"][0]["hv_trace"]) == 3  # 1 + 2 generations
    kinds = {candidate["kind"] for candidate in payload["per_candidate"]}
    assert kinds == {"controller", "alternative"}
    assert payload["summary"]["n_candidates"] == 3


def test_diagnosis_requires_a_problem(tmp_path: Path) -> None:
    """--problem is mandatory in diagnosis mode."""
    with pytest.raises(ValueError, match="--problem is required"):
        dhl.main(["--out", str(tmp_path / "x.json")])
    with pytest.raises(ValueError, match="generations"):
        dhl.main(["--problem", "zdt1", "--generations", "0"])


# --- report mode -------------------------------------------------------------


def _report_payload(
    *,
    generations: int = 20,
    n_states: int = 30,
    mean_percentile: float = 0.75,
    snr: float | None = 2.0,
    escape_50: dict[str, float | None] | None = None,
    escape_fractions: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A minimal diagnosis payload with controllable summary numbers."""
    escape_50 = (
        {"controller": 12.0, "default": 15.0, "alternative": 18.0}
        if escape_50 is None
        else escape_50
    )
    escape_fractions = (
        {"controller": 0.5, "default": 0.5, "alternative": 0.8}
        if escape_fractions is None
        else escape_fractions
    )
    return {
        "problem": "zdt4",
        "config": {"generations": generations, "n_reps": 3, "n_alternatives": 10,
                   "n_states_evaluated": n_states},
        "per_state": [
            {"state": f"zdt4|seed{1000 + index}|gen50", "best_minus_planner": 0.05}
            for index in range(n_states)
        ],
        "summary": {
            "n_states": n_states,
            "n_candidates": 12,
            "planner_action_rank_by_final_hv": {
                "mean_percentile": mean_percentile,
                "median_rank": 3.0,
                "n_states": n_states,
            },
            "final_hv_snr": {"mean": snr, "median": snr, "n_states": n_states},
            "mean_escape_gen_by_action_kind": {
                kind: {
                    "n_candidates": 10,
                    "mean_escape_gen_10pct": 5.0,
                    "mean_escape_gen_50pct": escape_50[kind],
                    "fraction_escaping_10pct": 0.9,
                    "fraction_escaping_50pct": escape_fractions[kind],
                }
                for kind in ("controller", "default", "alternative")
            },
        },
    }


def _verdict(report: dict[str, Any], cause: str) -> dict[str, Any]:
    matches = [entry for entry in report["verdicts"] if entry["cause"] == cause]
    assert len(matches) == 1, f"expected exactly one verdict for {cause}"
    return matches[0]


def test_report_judges_every_cause_with_numbers() -> None:
    """All four causes get a verdict with evidence and the numbers behind it."""
    report = dhl.assess_causes(_report_payload())
    causes = {entry["cause"] for entry in report["verdicts"]}
    assert {
        "wrong_target",
        "insufficient_data",
        "inadequate_action_space",
        "horizon_mismatch",
    } <= causes
    for entry in report["verdicts"]:
        assert entry["verdict"] in {
            dhl.VERDICT_SUPPORTED,
            dhl.VERDICT_NOT_SUPPORTED,
            dhl.VERDICT_INDISTINGUISHABLE,
        }
        assert entry["evidence"]
        assert isinstance(entry["numbers"], dict) and entry["numbers"]
    assert report["problem"] == "zdt4"
    assert report["caveats"]
    # a good planner percentile refutes the misaligned-target hypothesis
    good = _verdict(report, "wrong_target")
    assert good["verdict"] == dhl.VERDICT_NOT_SUPPORTED
    # SNR 2.0 sits between noise and a clear signal
    assert _verdict(report, "insufficient_data")["verdict"] == (
        dhl.VERDICT_INDISTINGUISHABLE
    )
    # some candidates escape -> the action space is adequate
    assert _verdict(report, "inadequate_action_space")["verdict"] == (
        dhl.VERDICT_NOT_SUPPORTED
    )
    # median escape 15 of 20 generations -> late, but not beyond 80%
    assert _verdict(report, "horizon_mismatch")["verdict"] == (
        dhl.VERDICT_INDISTINGUISHABLE
    )
    # a short branch cannot answer the long-horizon decoupling question
    assert _verdict(report, "wrong_target_long_horizon_decoupling")["verdict"] == (
        dhl.VERDICT_INDISTINGUISHABLE
    )


def test_report_rules_flip_with_the_data() -> None:
    """Each rule responds to the number it is built on."""
    poor_rank = dhl.assess_causes(_report_payload(mean_percentile=0.30))
    assert _verdict(poor_rank, "wrong_target")["verdict"] == dhl.VERDICT_SUPPORTED
    borderline = dhl.assess_causes(_report_payload(mean_percentile=0.55))
    assert _verdict(borderline, "wrong_target")["verdict"] == (
        dhl.VERDICT_INDISTINGUISHABLE
    )
    weak_signal = dhl.assess_causes(_report_payload(snr=0.6))
    assert _verdict(weak_signal, "insufficient_data")["verdict"] == (
        dhl.VERDICT_SUPPORTED
    )
    strong_signal = dhl.assess_causes(_report_payload(snr=6.2, n_states=30))
    assert _verdict(strong_signal, "insufficient_data")["verdict"] == (
        dhl.VERDICT_NOT_SUPPORTED
    )
    few_states = dhl.assess_causes(_report_payload(n_states=4, snr=5.0))
    assert _verdict(few_states, "insufficient_data")["verdict"] == (
        dhl.VERDICT_SUPPORTED
    )
    no_escape = dhl.assess_causes(
        _report_payload(escape_50={"controller": None, "default": None,
                                   "alternative": None})
    )
    assert _verdict(no_escape, "inadequate_action_space")["verdict"] == (
        dhl.VERDICT_INDISTINGUISHABLE
    )
    assert _verdict(no_escape, "horizon_mismatch")["verdict"] == (
        dhl.VERDICT_SUPPORTED
    )
    late_escape = dhl.assess_causes(
        _report_payload(escape_50={"controller": 19.0, "default": 19.0,
                                   "alternative": 18.0})
    )
    assert _verdict(late_escape, "horizon_mismatch")["verdict"] == (
        dhl.VERDICT_SUPPORTED
    )
    early_escape = dhl.assess_causes(
        _report_payload(escape_50={"controller": 4.0, "default": 5.0,
                                   "alternative": 6.0})
    )
    assert _verdict(early_escape, "horizon_mismatch")["verdict"] == (
        dhl.VERDICT_NOT_SUPPORTED
    )
    rare_escape = dhl.assess_causes(
        _report_payload(
            escape_50={"controller": 10.0, "default": 10.0, "alternative": 11.0},
            escape_fractions={"controller": 0.1, "default": 0.2, "alternative": 0.3},
        )
    )
    assert _verdict(rare_escape, "inadequate_action_space")["verdict"] == (
        dhl.VERDICT_SUPPORTED
    )
    # a 60-generation branch makes the long-horizon question observable
    long_branch = dhl.assess_causes(_report_payload(generations=60))
    assert not any(
        entry["cause"] == "wrong_target_long_horizon_decoupling"
        for entry in long_branch["verdicts"]
    )


def test_report_mode_reads_and_writes_files(diagnosis: dict[str, Any], tmp_path: Path) -> None:
    """--report consumes the diagnosis JSON and writes the report JSON."""
    report_out = tmp_path / "report.json"
    report = dhl.main(
        [
            "--report",
            "--out", str(diagnosis["out"]),
            "--report-out", str(report_out),
        ]
    )
    assert report_out.is_file()
    with report_out.open("r", encoding="utf-8") as fh:
        written = json.load(fh)
    assert written["problem"] == "zdt1"
    assert len(written["verdicts"]) >= 4
    assert set(written) >= {"problem", "config", "verdicts", "caveats"}
    assert report["problem"] == "zdt1"
    # the crafted payload is not mutated by the assessment
    crafted = _report_payload()
    snapshot = copy.deepcopy(crafted)
    dhl.assess_causes(crafted)
    assert crafted == snapshot
    with pytest.raises(FileNotFoundError, match="diagnosis file not found"):
        dhl.main(["--report", "--out", str(tmp_path / "absent.json")])


def test_cli_defaults() -> None:
    """CLI defaults match the documented diagnosis grid."""
    args = dhl.parse_args(["--problem", "zdt4"])
    assert args.problem == "zdt4"
    assert args.out == dhl.DEFAULT_OUT
    assert args.report_out == dhl.DEFAULT_REPORT_OUT
    assert args.generations == dhl.DEFAULT_GENERATIONS == 20
    assert args.max_states == dhl.DEFAULT_MAX_STATES == 30
    assert args.n_alternatives == dhl.DEFAULT_N_ALTERNATIVES == 10
    assert args.n_reps == dhl.DEFAULT_N_REPS == 3
    assert args.include_default_action is True
    assert args.controller_type == "planning"
    assert args.stagnation_epsilon == dhl.DEFAULT_STAGNATION_EPSILON
    assert args.exploration_std_fraction == dhl.DEFAULT_EXPLORATION_STD_FRACTION
    assert args.report is False and args.store_per_rep is False
    assert dhl.ESCAPE_THRESHOLDS == (0.10, 0.50)
    assert (
        dhl.parse_args(["--problem", "zdt4", "--no-include-default-action"])
        .include_default_action
        is False
    )
    assert dhl.parse_args(["--report"]).report is True
