"""Stage transitions, plateau evidence and accepted cost accounting."""

from dataclasses import asdict, dataclass, field


@dataclass
class StageProgress:
    stages: tuple[str, ...]
    index: int = 0
    without_accept: int = 0
    ends: dict = field(default_factory=dict)
    cost_delta: dict = field(default_factory=dict)

    @property
    def stage(self):
        return self.stages[self.index] if self.index < len(self.stages) else "finalization"

    def advance(self, reason):
        self.ends[self.stage] = reason
        self.index += 1
        self.without_accept = 0

    def rejected(self, stage):
        if stage != "cleanup":
            self.without_accept += 1

    def accepted(self, stage, feedback):
        deltas = self.cost_delta.setdefault(stage, dict.fromkeys(asdict(feedback.base_cost), 0))
        for name, before in asdict(feedback.base_cost).items():
            deltas[name] += before - getattr(feedback.candidate_cost, name)
        self.without_accept = 0
