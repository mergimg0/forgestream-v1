"""Dashboard launcher -- creates a FastAPI app with optional Firestore connection."""

from __future__ import annotations

import logging

from ..config import ForgeStreamConfig
from .server import create_app

logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    firebase_admin = None  # type: ignore
    firestore = None  # type: ignore


def create_live_app(config: ForgeStreamConfig):
    """Create the dashboard app, optionally connected to Firestore."""
    db = None

    if config.firestore_enabled and HAS_FIREBASE:
        try:
            try:
                firebase_admin.get_app()
            except ValueError:
                import os
                # ot-ctx-gcp-cred-003: resolve relative to project root
                _proj_root = os.path.join(os.path.dirname(__file__), "..", "..")
                sa_path = os.environ.get(
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    os.path.join(_proj_root, ".secrets", "service-account.json"),
                )
                if os.path.exists(sa_path):
                    cred = credentials.Certificate(sa_path)
                else:
                    cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(
                    credential=cred,
                    options={"projectId": config.firebase_project},
                )
            db = firestore.client()
            logger.info("Dashboard connected to Firestore (project: %s)", config.firebase_project)
        except Exception as e:
            logger.warning("Dashboard Firestore connection failed: %s", e)

    return create_app(firestore_db=db)
