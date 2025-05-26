from setuptools import setup, Extension, find_packages
import numpy as np

module = Extension('bitarray_heapsort',
                  sources = ['bitarray_heapsort.c'],
                  include_dirs=[np.get_include()])

setup(
    # Métadonnées du projet global
    name = 'strainminer',
    version = '0.1.0',
    description = 'Outil pour l\'analyse des souches microbiennes',
    
    # Configuration du package Python
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Extension C existante
    ext_modules = [module],
    
    # Dépendances
    install_requires=[
        'numpy',
        # ajoutez ici vos autres dépendances
    ],
    
    # Point d'entrée pour la ligne de commande
    entry_points={
        'console_scripts': [
            'strainminer=strainminer.__main__:init',
        ],
    },
)