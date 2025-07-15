"""
DR.SIMON: Domain-wise Rewrite for Segment-Informed Medical Oversight Network

A query-rewriting framework for temporal grounding in medical videos.
"""

from .modules.qrm import QueryRewritingModule
from .modules.besm import BoundaryAwareEventSegmentationModule  
from .modules.qem import QueryEventMatchingModule

__all__ = [
    "QueryRewritingModule",
    "BoundaryAwareEventSegmentationModule", 
    "QueryEventMatchingModule"
] 