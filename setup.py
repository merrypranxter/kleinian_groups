from setuptools import setup, find_packages

setup(
    name='kleinian',
    version='0.1.0',
    description='Kleinian groups and hyperbolic geometry in Python',
    author='Kleinian Groups Project',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'Pillow>=8.3.0',
    ],
    extras_require={
        'dev': ['pytest>=6.2.0', 'jupyter>=1.0.0'],
    },
)
