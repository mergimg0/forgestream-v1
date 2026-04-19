"""Audio emotion detection pipeline for ForgeStream."""
from .buffer import AudioRingBuffer
from .correlator import EmotionCorrelator
from .crqa_router import CRQAComputeRouter, CRQAResult
from .disengagement import DisengagementDetector
from .dynamics import GroupDynamicsEngine
from .extractor import EmotionExtractor
from .features import EGeMAPSExtractor, PraatExtractor, ProsodicFeatures
from .persistence import EmotionCorpus
from .rapport import RapportEngine, interpolate_weights
from .speaker import SpeakerTimeSeries
from .transfer_entropy import compute_symmetry, compute_transfer_entropy

__all__ = [
    "AudioRingBuffer",
    "CRQAComputeRouter",
    "CRQAResult",
    "DisengagementDetector",
    "EGeMAPSExtractor",
    "EmotionCorpus",
    "EmotionCorrelator",
    "EmotionExtractor",
    "GroupDynamicsEngine",
    "PraatExtractor",
    "ProsodicFeatures",
    "RapportEngine",
    "SpeakerTimeSeries",
    "compute_symmetry",
    "compute_transfer_entropy",
    "interpolate_weights",
]
