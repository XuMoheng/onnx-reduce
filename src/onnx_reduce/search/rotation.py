"""Fair pass rotation and bounded progress tracking for staged search."""

from collections import deque


class PassRotation:
    def __init__(self):
        self.turn_pass = 0
        self.turn_used = 0
        self.turn_budget = None
        self.pass_history = {}
        self.pass_failures = {}
        self.pass_states = {}

    def quantum(self, index, quantum):
        history = self.pass_history.get(index, ())
        # Deterministic success feedback, never wall-clock timings. Every pass
        # retains an eight-attempt turn; productive passes get at most sixteen.
        return quantum * (2 if len(history) >= 8 and sum(history) * 4 >= len(history) else 1)

    def observe(self, attempt, feedback, pass_count, quantum):
        if attempt.stage == "cleanup":
            return
        if self.turn_pass != attempt.pass_index:
            self.turn_used = 0
            self.turn_budget = None
        if self.turn_budget is None:
            self.turn_budget = self.quantum(attempt.pass_index, quantum)
        self.pass_history.setdefault(attempt.pass_index, deque(maxlen=16)).append(
            feedback.status == "accepted"
        )
        self.turn_pass = attempt.pass_index
        self.turn_used += 1
        if self.turn_used >= self.turn_budget:
            self.turn_pass = (attempt.pass_index + 1) % max(1, pass_count)
            self.turn_used = 0
            self.turn_budget = None
        key = (attempt.stage, attempt.pass_index)
        self.pass_failures[key] = self.pass_failures.get(key, 0) + 1

    def accepted(self):
        self.pass_failures.clear()
        self.pass_states.clear()

    def candidates(self, model, stage, *, passes, generate, quantum, limit):
        indices = list(range(len(passes)))
        start = self.turn_pass if stage != "cleanup" else 0
        indices = indices[start:] + indices[:start]
        queue = deque((index, iter(generate(model, stage, index))) for index in indices)
        for index in indices:
            self.pass_states.setdefault(f"{stage}/{passes[index].name}", "not_visited")
        while queue:
            index, stream = queue.popleft()
            key = (stage, index)
            if self.pass_failures.get(key, 0) >= limit:
                self.pass_states[f"{stage}/{passes[index].name}"] = "plateau"
                continue
            state_key = f"{stage}/{passes[index].name}"
            yielded = False
            turn_quantum = self.quantum(index, quantum)
            if index == self.turn_pass and stage != "cleanup":
                turn_quantum = max(1, (self.turn_budget or turn_quantum) - self.turn_used)
            for _ in range(turn_quantum):
                if self.pass_failures.get(key, 0) >= limit:
                    break
                attempt = next(stream, None)
                if attempt is None:
                    self.pass_states[state_key] = (
                        "exhausted"
                        if yielded or self.pass_states.get(state_key) == "visited"
                        else "no_candidates"
                    )
                    break
                yielded = True
                self.pass_states[state_key] = "visited"
                yield attempt
            else:
                queue.append((index, stream))
                continue
            if self.pass_failures.get(key, 0) >= limit:
                self.pass_states[state_key] = "plateau"
