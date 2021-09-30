# Copyright 2021 AI Singapore. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='data analysis model prediction diagnostic',
    author='Yap Siew Lin, Jeanne Choo, Chong Wei Yih @ AI Singapore',
    author_email='siewlin@aisingapore.org',
    license='Apache 2.0',
    url='https://github.com/aimakerspace/Rarity',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data={'rarity': ['py.typed']},
    entry_points={'console_scripts': ['rarity-cli=rarity.commands.rarity_cli:main']},
    python_requires='>=3.6.0',
    install_requires=REQUIRED_PKGS,
    extras_require={'notebook': ['jupyter-client>=6.0.0', 'jupyter-core>=4.6.3', 'ipywidgets>=7.5.1']},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
