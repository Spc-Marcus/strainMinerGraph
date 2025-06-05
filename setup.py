from setuptools import setup, Extension, find_packages
import numpy

# Module d'extension avec OpenMP support
bitarray_module = Extension(
    'bitarray_heapsort',
    sources=['src/utils/bitarray_heapsort.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
    extra_link_args=['-fopenmp'],
)

# Configuration de base
setup(
    name='strainminer',
    version='2.1.0',
    description='Outil pour l\'analyse des souches microbiennes',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[bitarray_module],
    install_requires=[
        'numpy',
        'matplotlib',
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
        ],
    },
    zip_safe=False,
)