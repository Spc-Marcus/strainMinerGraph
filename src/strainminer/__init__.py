"""
StrainMiner - Un outil pour l'analyse des souches microbiennes
"""

# Utilisez des importations relatives pour éviter les problèmes
from .core import (
    __version__,
    file_path,
    get_data
)

# Évitez l'importation circulaire en important init à l'utilisation
__all__ = [
    '__version__',
    'file_path',
    'get_data'
]