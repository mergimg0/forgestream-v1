from forgestream.synthesis.suggestions import Priority, Suggestion, SuggestionQueue


class TestPriority:
    def test_priority_categories(self):
        assert Priority.from_score(0.95) == Priority.CRITICAL
        assert Priority.from_score(0.75) == Priority.STRATEGIC
        assert Priority.from_score(0.55) == Priority.DELVE_DEEPER
        assert Priority.from_score(0.35) == Priority.GOOD_TO_PROBE
        assert Priority.from_score(0.15) == Priority.NICE_TO_KNOW


class TestSuggestionQueue:
    def test_add_and_get_top(self):
        q = SuggestionQueue()
        q.add(Suggestion(text="low priority", priority_score=0.2))
        q.add(Suggestion(text="high priority", priority_score=0.9))
        q.add(Suggestion(text="medium", priority_score=0.5))

        top = q.peek()
        assert top is not None
        assert top.text == "high priority"

    def test_dismiss_removes_top(self):
        q = SuggestionQueue()
        q.add(Suggestion(text="top", priority_score=0.9))
        q.add(Suggestion(text="second", priority_score=0.5))

        q.dismiss()
        top = q.peek()
        assert top is not None
        assert top.text == "second"

    def test_decay_reduces_scores(self):
        q = SuggestionQueue()
        s = Suggestion(text="decaying", priority_score=0.8, decay_rate=0.1)
        q.add(s)
        q.apply_decay(steps=3)
        top = q.peek()
        assert top is not None
        assert top.priority_score < 0.8

    def test_get_all_by_priority(self):
        q = SuggestionQueue()
        q.add(Suggestion(text="a", priority_score=0.95))
        q.add(Suggestion(text="b", priority_score=0.75))
        q.add(Suggestion(text="c", priority_score=0.15))

        critical = q.get_by_priority(Priority.CRITICAL)
        assert len(critical) == 1

    def test_len(self):
        q = SuggestionQueue()
        assert len(q) == 0
        q.add(Suggestion(text="x", priority_score=0.5))
        assert len(q) == 1

    def test_empty_peek_returns_none(self):
        q = SuggestionQueue()
        assert q.peek() is None
