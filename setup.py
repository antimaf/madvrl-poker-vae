from setuptools import setup, find_packages

setup(
    name='madvrl-poker-vae',
    version='0.1.0',
    description='Variational Autoencoder for Poker Game State Inference',
    author='Anthony Imafidon',
    author_email='anthonyimafidon24@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.8.0',
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.60.0',
        'pyarrow>=4.0.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    python_requires='>=3.8'
)
