from setuptools import setup, find_packages

setup(
    name='madvrl',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'tqdm',
        'pyarrow'
    ],
    author='Your Name',
    description='Multi-Agent Deep Variational Reinforcement Learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
