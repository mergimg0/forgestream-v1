"""Cross-meeting transfer tests."""

from uuid import uuid4

from forgestream.events.schema import Event, EventType
from forgestream.graph.materializer import GraphMaterializer
from forgestream.graph.transfer import CrossMeetingTransfer, SeedGarden


class TestCrossMeetingTransfer:
    def _make_session_events(self, session_id, keywords_list):
        bid = uuid4()
        events = []
        for keywords in keywords_list:
            events.append(Event(
                event_type=EventType.CLAIM,
                session_id=session_id,
                branch_id=bid,
                author="gemini",
                evaluator=0.5,
                payload={"text": "claim", "topic_keywords": keywords, "confidence": 0.8},
            ))
        return events

    def test_merge_graphs(self):
        transfer = CrossMeetingTransfer()
        materializer = GraphMaterializer()

        s1 = uuid4()
        s2 = uuid4()
        events1 = self._make_session_events(s1, [["Kafka", "ingestion"], ["pipeline", "data"]])
        events2 = self._make_session_events(s2, [["Flink", "streaming"], ["pipeline", "realtime"]])

        g1 = materializer.materialize(events1)
        g2 = materializer.materialize(events2)

        merged = transfer.merge_graphs([g1, g2])
        assert merged.get_concept("Kafka") is not None
        assert merged.get_concept("Flink") is not None
        assert merged.get_concept("pipeline") is not None

    def test_detect_shared_concepts(self):
        transfer = CrossMeetingTransfer()
        materializer = GraphMaterializer()

        s1 = uuid4()
        s2 = uuid4()
        events1 = self._make_session_events(s1, [["pipeline", "data"], ["latency"]])
        events2 = self._make_session_events(s2, [["pipeline", "streaming"], ["throughput"]])

        g1 = materializer.materialize(events1)
        g2 = materializer.materialize(events2)

        shared = transfer.find_shared_concepts(g1, g2)
        assert "pipeline" in shared

    def test_no_shared_concepts(self):
        transfer = CrossMeetingTransfer()
        materializer = GraphMaterializer()

        s1 = uuid4()
        s2 = uuid4()
        events1 = self._make_session_events(s1, [["Kafka", "ingestion"]])
        events2 = self._make_session_events(s2, [["React", "frontend"]])

        g1 = materializer.materialize(events1)
        g2 = materializer.materialize(events2)

        shared = transfer.find_shared_concepts(g1, g2)
        assert len(shared) == 0


class TestSeedGarden:
    def test_add_and_list_seeds(self):
        garden = SeedGarden()
        garden.add_seed(
            session_id="meeting-1",
            cluster_nodes=["quantum", "entanglement"],
            novelty_score=0.8,
        )
        garden.add_seed(
            session_id="meeting-2",
            cluster_nodes=["blockchain", "consensus"],
            novelty_score=0.6,
        )
        seeds = garden.list_seeds()
        assert len(seeds) == 2

    def test_promote_seed(self):
        garden = SeedGarden()
        seed_id = garden.add_seed(
            session_id="m1",
            cluster_nodes=["X", "Y"],
            novelty_score=0.9,
        )
        garden.promote(seed_id)
        seed = garden.get_seed(seed_id)
        assert seed["status"] == "promoted"

    def test_archive_seed(self):
        garden = SeedGarden()
        seed_id = garden.add_seed(
            session_id="m1",
            cluster_nodes=["A"],
            novelty_score=0.3,
        )
        garden.archive(seed_id)
        seed = garden.get_seed(seed_id)
        assert seed["status"] == "archived"
