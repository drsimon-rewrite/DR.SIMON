"""
DR.SIMON Modules

This package contains the three main modules of DR.SIMON:
- QRM: Query Rewriting Module
- BESM: Boundary-aware Event Segmentation Module  
- QEM: Query-Event Matching Module
"""

from .qrm import QueryRewritingModule
from .besm import BoundaryAwareEventSegmentationModule
from .qem import QueryEventMatchingModule

__all__ = [
    "QueryRewritingModule",
    "BoundaryAwareEventSegmentationModule",
    "QueryEventMatchingModule"
] 