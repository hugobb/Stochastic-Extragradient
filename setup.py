import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hamiltonian",
    version="0.1",
    author="Hugo Berard",
    author_email="berard.hugo@gmail.com",
    description="Stochastic Gradient Descent-Ascent and Consensus Optimization for Smooth Games: Convergence Analysis under Expected Co-coercivity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hugobb/StochasticGamesOpt",
    project_urls={
        "Bug Tracker": "https://github.com/hugobb/StochasticGamesOpt/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "hamiltonian"},
    packages=setuptools.find_packages(where="hamiltonian"),
    python_requires=">=3.6",
    install_requires=[
        'tqdm',
        'numpy',
        'torch',
        'scipy',
      ],
)
