from dataclasses import dataclass


@dataclass(frozen=True)
class UtoMeStrategy:
    """Base description of an uncertainty-aware token merging variant."""

    name: str
    needs_external_uq: bool
    learns_uq_predictor: bool
    teacher_uq_source: str
    uses_direct_teacher_for_merge: bool


UTOME_STRATEGIES = {
    "utome_e": UtoMeStrategy(
        name="utome_e",
        needs_external_uq=True,
        learns_uq_predictor=False,
        teacher_uq_source="eda",
        uses_direct_teacher_for_merge=True,
    ),
    "utome_p": UtoMeStrategy(
        name="utome_p",
        needs_external_uq=True,
        learns_uq_predictor=True,
        teacher_uq_source="eda",
        uses_direct_teacher_for_merge=False,
    ),
    "utome_s": UtoMeStrategy(
        name="utome_s",
        needs_external_uq=False,
        learns_uq_predictor=True,
        teacher_uq_source="proxy",
        uses_direct_teacher_for_merge=False,
    ),
}

# Rebuttal-only matched-budget token-pruning baselines. `prune` ranks tokens with
# the same teacher score as UtoMe (so `--override-teacher-mode` selects between
# dynamics-saliency pruning, uncertainty-only pruning, and UtoMe-score pruning);
# `prune_random` drops uniformly at random.
UTOME_STRATEGIES["prune"] = UtoMeStrategy(
    name="prune",
    needs_external_uq=False,
    learns_uq_predictor=False,
    teacher_uq_source="eda",
    uses_direct_teacher_for_merge=True,
)
UTOME_STRATEGIES["prune_random"] = UtoMeStrategy(
    name="prune_random",
    needs_external_uq=False,
    learns_uq_predictor=False,
    teacher_uq_source="eda",
    uses_direct_teacher_for_merge=False,
)

PRUNE_METHODS = ("prune", "prune_random")

UTOME_STRATEGIES["utome"] = UTOME_STRATEGIES["utome_e"]
UTOME_STRATEGIES["utome_plus"] = UTOME_STRATEGIES["utome_p"]
UTOME_STRATEGIES["utome_pp"] = UTOME_STRATEGIES["utome_s"]


def get_utome_strategy(method: str) -> UtoMeStrategy | None:
    return UTOME_STRATEGIES.get(method)
