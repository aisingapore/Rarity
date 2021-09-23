from setuptools import setup, find_packages


REQUIRED_PKGS = [
    "pandas>=1.2.4",
    "scikit-learn>=0.24.1",
    "statsmodels==0.12.2",
    "plotly>=4.14.3",
    "dash==1.20.0",
    "dash-bootstrap-components==0.12.2",
    "dash-core-components==1.16.0",
    "dash-html-components==1.1.3",
]

setup(
    name='rarity',
    version='1.0.dev0',
    description='Data diagnostic package with minimum setup to analyze miss predictions of ML models',
    keywords='data-analysis model-prediction dianogstic',
    author='Yap Siew Lin, Jeanne Choo, Chong Wei Yih @ AI Singapore',
    author_email='siewlin@aisingapore.org',
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": ["rarity-cli=rarity.commands.rarity_cli:main"]},
    python_requires=">=3.6.0",
    install_requires=REQUIRED_PKGS,
    extras_require={"notebook": ["jupyter-client>=6.0.0", "jupyter-core>=4.6.3", "ipywidgets>=7.5.1"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
