"""Group inference for subject, node, and system summaries."""
from .inference import InferenceResult, bh_fdr, infer_groups

__all__ = ["InferenceResult", "bh_fdr", "infer_groups"]
