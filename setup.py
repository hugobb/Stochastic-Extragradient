import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gamesopt",
    version="0.1",
    author="Hugo Berard",
    author_email="berard.hugo@gmail.com",
    description="Stochastic Extragradient: General Analysis and Improved Rates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hugobb/SEG_AISTATS_2022",
    project_urls={
        "Bug Tracker": "https://github.com/hugobb/SEG_AISTATS_2022/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "gamesopt"},
    packages=setuptools.find_packages(where="gamesopt"),
    python_requires=">=3.6",
    install_requires=[
        'tqdm',
        'numpy',
        'torch',
        'scipy',
      ],
)
