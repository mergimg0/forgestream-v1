"""FastAPI dashboard server."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import create_router

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_app(firestore_db: Any = None) -> FastAPI:
    """Create the dashboard FastAPI application."""
    app = FastAPI(title="ForgeStream Dashboard")
    router = create_router(firestore_db=firestore_db)
    app.include_router(router, prefix="/api")

    # Mount static files (CSS, JS)
    if os.path.isdir(_STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ForgeStream Dashboard</title>
    <link rel="stylesheet" href="/static/css/dashboard.css">
</head>
<body>
    <h1><span class="status-dot"></span>ForgeStream Dashboard</h1>

    <div class="dashboard-grid">

        <!-- Row 1: Knowledge Graph (full width) -->
        <div class="panel panel-knowledge-graph">
            <div class="panel-header">
                <h2>Knowledge Graph</h2>
                <span class="panel-badge">force-directed</span>
            </div>
            <div id="knowledge-graph"></div>
        </div>

        <!-- Row 2 left: Evaluator Trajectory -->
        <div class="panel">
            <div class="panel-header">
                <h2>Evaluator Trajectory</h2>
                <span class="panel-badge">E(&pi;)</span>
            </div>
            <div id="evaluator-trajectory"></div>
        </div>

        <!-- Row 2 right: Entrainment Heatmap -->
        <div class="panel">
            <div class="panel-header">
                <h2>Entrainment</h2>
                <span class="panel-badge">speaker correlation</span>
            </div>
            <div id="entrainment-heatmap"></div>
        </div>

        <!-- Row 3: Meeting Timeline (full width) -->
        <div class="panel panel-timeline">
            <div class="panel-header">
                <h2>Meeting Timeline</h2>
                <span class="panel-badge">events</span>
            </div>
            <div id="meeting-timeline"></div>
        </div>

        <!-- Row 4 left: Emotion Timeline -->
        <div class="panel">
            <div class="panel-header">
                <h2>Emotion Timeline</h2>
                <span class="panel-badge">per-speaker</span>
            </div>
            <div id="emotion-timeline"></div>
        </div>

        <!-- Row 4 right: Rapport Trajectory -->
        <div class="panel">
            <div class="panel-header">
                <h2>Rapport + Engagement</h2>
                <span class="panel-badge">composite</span>
            </div>
            <div id="rapport-trajectory"></div>
        </div>

        <!-- Row 5 left: Branch Tree -->
        <div class="panel">
            <div class="panel-header">
                <h2>Branch Tree</h2>
                <span class="panel-badge">conversation drift</span>
            </div>
            <div id="branch-tree"></div>
        </div>

        <!-- Row 5 right: Seed Garden -->
        <div class="panel">
            <div class="panel-header">
                <h2>Seed Garden</h2>
                <span class="panel-badge">latent topics</span>
            </div>
            <div id="seed-garden"></div>
        </div>

        <!-- Row 6 left: SOS Convergence -->
        <div class="panel">
            <div class="panel-header">
                <h2>SOS Convergence</h2>
                <span class="panel-badge">&epsilon; trust region</span>
            </div>
            <div id="sos-convergence"></div>
        </div>

        <!-- Row 6 right: Contradictions -->
        <div class="panel">
            <div class="panel-header">
                <h2>Contradictions</h2>
                <span class="panel-badge">resolution queue</span>
            </div>
            <div id="contradictions"></div>
        </div>

        <!-- Row 7: Autonomy Progression + Proof Queue -->
        <div class="panel">
            <div class="panel-header">
                <h2>Autonomy Progression</h2>
                <span class="panel-badge">&epsilon; trajectory</span>
            </div>
            <div id="autonomy-progression"></div>
        </div>

        <!-- Row 7 right: Proof Queue (full width) -->
        <div class="panel panel-proof-queue">
            <div class="panel-header">
                <h2>Proof Queue</h2>
                <span class="panel-badge">Lean 4 obligations</span>
            </div>
            <div id="proof-queue"></div>
        </div>

    </div>

    <div class="refresh-indicator">
        <span class="status-dot"></span>
        auto-refresh 5s
    </div>

    <!-- D3 v7 from CDN -->
    <script src="https://d3js.org/d3.v7.min.js"></script>

    <!-- Panel modules -->
    <script src="/static/js/knowledge-graph.js"></script>
    <script src="/static/js/evaluator-trajectory.js"></script>
    <script src="/static/js/meeting-timeline.js"></script>
    <script src="/static/js/emotion-timeline.js"></script>
    <script src="/static/js/entrainment-heatmap.js"></script>
    <script src="/static/js/rapport-trajectory.js"></script>
    <script src="/static/js/branch-tree.js"></script>
    <script src="/static/js/seed-garden.js"></script>
    <script src="/static/js/sos-convergence.js"></script>
    <script src="/static/js/contradictions.js"></script>
    <script src="/static/js/proof-queue.js"></script>
    <script src="/static/js/autonomy-progression.js"></script>

    <script>
        // Instantiate panels
        const knowledgeGraph       = new KnowledgeGraph('knowledge-graph');
        const evaluatorTrajectory  = new EvaluatorTrajectory('evaluator-trajectory');
        const meetingTimeline      = new MeetingTimeline('meeting-timeline');
        const emotionTimeline      = new EmotionTimeline('emotion-timeline');
        const entrainmentHeatmap   = new EntrainmentHeatmap('entrainment-heatmap');
        const rapportTrajectory    = new RapportTrajectory('rapport-trajectory');
        const branchTree           = new BranchTree('branch-tree');
        const seedGarden           = new SeedGarden('seed-garden');
        const sosConvergence       = new SOSConvergence('sos-convergence');
        const contradictions       = new Contradictions('contradictions');
        const proofQueue           = new ProofQueue('proof-queue');
        const autonomyProgression  = new AutonomyProgression('autonomy-progression');

        // Initial render
        knowledgeGraph.update();
        evaluatorTrajectory.update();
        meetingTimeline.update();
        emotionTimeline.update();
        entrainmentHeatmap.update();
        rapportTrajectory.update();
        branchTree.update();
        seedGarden.update();
        sosConvergence.update();
        contradictions.update();
        proofQueue.update();
        autonomyProgression.update();

        // Auto-refresh every 5 seconds
        setInterval(() => {
            knowledgeGraph.update();
            evaluatorTrajectory.update();
            meetingTimeline.update();
            emotionTimeline.update();
            entrainmentHeatmap.update();
            rapportTrajectory.update();
            branchTree.update();
            seedGarden.update();
            sosConvergence.update();
            contradictions.update();
            proofQueue.update();
            autonomyProgression.update();
        }, 5000);
    </script>
</body>
</html>"""
