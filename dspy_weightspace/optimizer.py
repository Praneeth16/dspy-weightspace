"""WeightSpaceOptimizer: a DSPy Teleprompter whose search space is the student LM's weights.

Three rungs of the same ``compile()`` interface:
  dspy.GEPA              edits instruction text        (LLM calls)
  WeightSpaceOptimizer   edits a sparse weight patch   (LLM calls + a few forward passes, reversible diff)
  dspy.BootstrapFinetune edits all weights             (GPU training job)
"""

from __future__ import annotations

import os

import dspy
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.annotation import experimental
from gepa import optimize

from .adapter import COMPONENT, WeightSpaceAdapter
from .hf_lm import HFLocalLM
from .probes import Probes
from .program import MathSolver
from .proposer import WeightPatchProposer
from .sandbox import run_fresh


@experimental(version="0.1.0")
class WeightSpaceOptimizer(Teleprompter):
    """Let a frontier reflection LM read, probe and patch the weights of a small local model used by a DSPy program.

    Args:
        metric: GEPA-shaped metric ``(gold, pred, trace, pred_name, pred_trace) -> Prediction(score, feedback)``.
        reflection_lm: the frontier model that reasons about weights (any ``dspy.LM``).
        max_metric_calls: GEPA budget in program evaluations. Defaults to ``4 * len(valset)``.
        passes_per_round: forward-pass budget the reflection LM gets per proposal round.
        reflection_minibatch_size: examples shown to the reflection LM per round.
        seed_patch: optional starting patch script (for example the published Experiment 8 edit).
        log_dir: transcripts, probe outputs, artifacts and GEPA state.

    ``compile(trainset, valset)``: the reflection LM probes and reads feedback on ``trainset`` only; GEPA selects
    candidates on ``valset``. Keep them disjoint, and keep the final test set out of both: with one set for feedback and
    selection, an edit can win on the examples it was written against and lose on new ones. Without a ``valset`` the
    last two thirds of ``trainset`` become the selection set.
    """

    def __init__(
        self,
        metric,
        reflection_lm: dspy.LM,
        *,
        max_metric_calls: int | None = None,
        passes_per_round: int = 8,
        max_items_per_pass: int = 32,
        reflection_minibatch_size: int = 8,
        max_turns: int = 40,
        seed_patch: str = "",
        log_dir: str = "runs/weightspace",
        seed: int = 0,
    ):
        super().__init__()
        self.metric, self.reflection_lm = metric, reflection_lm
        self.max_metric_calls, self.passes_per_round, self.max_items_per_pass = (
            max_metric_calls,
            passes_per_round,
            max_items_per_pass,
        )
        self.reflection_minibatch_size, self.max_turns = reflection_minibatch_size, max_turns
        self.seed_patch, self.log_dir, self.seed = seed_patch, log_dir, seed

    def compile(self, student: MathSolver, *, trainset: list[dspy.Example], teacher=None, valset=None) -> MathSolver:
        if teacher is not None:
            raise ValueError("WeightSpaceOptimizer does not use a teacher")
        lm = getattr(student, "lm", None)
        if not isinstance(lm, HFLocalLM):
            raise ValueError("student.lm must be an HFLocalLM (in-process weights are required to patch them)")
        if valset is None:
            cut = max(1, len(trainset) // 3)
            trainset, valset = trainset[:cut], trainset[cut:]
        os.makedirs(self.log_dir, exist_ok=True)

        probes = Probes(
            lm,
            student,
            self.metric,
            trainset,
            os.path.join(self.log_dir, "probes"),
            budget=self.passes_per_round,
            max_items=self.max_items_per_pass,
        )
        proposer = WeightPatchProposer(self.reflection_lm, lm, probes, self.log_dir, max_turns=self.max_turns)
        adapter = WeightSpaceAdapter(lm, student, self.metric, self.log_dir, proposer)

        result = optimize(
            seed_candidate={COMPONENT: self.seed_patch},
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            candidate_selection_strategy="pareto",
            reflection_minibatch_size=self.reflection_minibatch_size,
            module_selector="round_robin",
            use_merge=False,
            max_metric_calls=self.max_metric_calls or 4 * len(valset),
            run_dir=os.path.join(self.log_dir, "gepa"),
            seed=self.seed,
            raise_on_exception=False,
            display_progress_bar=True,
        )

        best = result.best_candidate[COMPONENT]
        lm.reset()
        sb, res = run_fresh(lm, best, self.log_dir)
        if res.ok:
            sb.apply()
        student.patch_src = best
        student.detailed_results = result
        student.weights_changed = lm.changed_entries()
        student.edited_tensors = lm.edited()
        student._compiled = True
        return student
