from setuptools import setup, Extension, find_packages

# Fonction pour obtenir les chemins d'inclusion numpy de façon sécurisée
def get_numpy_include():
    try:
        import numpy
        return numpy.get_include()
    except ImportError:
        return None

# Module d'extension avec les en-têtes numpy
module = Extension('bitarray_heapsort',
                  sources = ['src/utils/bitarray_heapsort.c'],
                  include_dirs=[get_numpy_include()] if get_numpy_include() else [])

# Configuration de base
setup(
    name = 'strainminer',
    version = '1.1.0',
    description = 'Outil pour l\'analyse des souches microbiennes',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules = [module],
    install_requires=[
        'numpy',
        'PySide6',
    ],
    setup_requires=[
        'numpy',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
        ],
    },
    entry_points={
        'console_scripts': [
            'strainminer=strainminer.__main__:main',
            'strainminer-gui=strainminer.ui:run_app',
        ],
    },
)