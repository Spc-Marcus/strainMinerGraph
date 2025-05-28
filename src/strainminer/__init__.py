"""
StrainMiner - Microbial strain analysis tool
Based on the work of Tam Khac Min Truong and Roland Faure
Author: FOIN Marcus
"""

from .core import __version__
from .pipeline import StrainMinerPipeline, run_strainminer_pipeline

# Public API
__all__ = [
    '__version__',
    'StrainMinerPipeline', 
    'run_strainminer_pipeline'
]