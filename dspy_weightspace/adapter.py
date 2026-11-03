"""GEPA adapter whose single component is a weight patch script, not an instruction."""

from __future__ import annotations

import dspy
from gepa.core.adapter import EvaluationBatch

from .hf_lm import HFLocalLM
from .program import MathSolver
from .sandbox import run_fresh

COMPONENT = "weight_patch"


class WeightSpaceAdapter:
    def __init__(self, lm: HFLocalLM, program: MathSolver, metric, work_dir: str, proposer, failure_score: float = 0.0):
        self.lm, self.program, self.metric, self.work_dir = lm, program, metric, work_dir
        self.propose_new_texts = proposer
        self.failure_score = failure_score
        self.patch_cache: dict[str, str | None] = {}

    def evaluate(
        self, batch: list[dspy.Example], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        src = candidate.get(COMPONENT, "")
        sb, res = run_fresh(self.lm, src, self.work_dir)
        if not res.ok:
            self.lm.reset()
            traj = [
                {"example": ex, "prediction": None, "feedback": f"patch script failed: {res.error}"} for ex in batch
            ]
            return EvaluationBatch(
                outputs=[None] * len(batch),
                scores=[self.failure_score] * len(batch),
                trajectories=traj if capture_traces else None,
                num_metric_calls=len(batch),
            )
        sb.apply()
        try:
            preds = self.program.batch_forward([ex.problem for ex in batch], seed=0)
        finally:
            self.lm.reset()
        scores, traj = [], []
        for ex, pred in zip(batch, preds, strict=True):
            m = self.metric(ex, pred)
            scores.append(float(m.score))
            traj.append({"example": ex, "prediction": pred, "feedback": m.feedback})
        return EvaluationBatch(
            outputs=preds, scores=scores, trajectories=traj if capture_traces else None, num_metric_calls=len(batch)
        )

    def make_reflective_dataset(self, candidate, eval_batch: EvaluationBatch, components_to_update):
        rows = []
        order = sorted(range(len(eval_batch.scores)), key=lambda i: eval_batch.scores[i])
        for i in order:
            t = eval_batch.trajectories[i]
            pred = t["prediction"]
            rows.append(
                {
                    "Inputs": {"problem": t["example"].problem, "answer": t["example"].answer},
                    "Generated Outputs": (pred.text if pred is not None else ""),
                    "Feedback": t["feedback"],
                }
            )
        return {COMPONENT: rows}
