import os
import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit('Requires Python 3.6 or higher')

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join('README.md'), encoding='utf-8') as f:
    readme = f.read()

REQUIRED_PACKAGES = [
    'tensorflow-gpu == 1.15.4',
    'matplotlib == 3.1.1',
    'numpy == 1.14.5',
    'pandas == 0.25.3',
    'plotly == 4.2.1',
    'SimpleITK == 1.2.3',
    'scikit-image == 0.16.2',
    'pymia == 0.2.2',
    'torch == 1.3.1',  # only used for data loading
    'tensorboardX == 1.9',
]

TEST_PACKAGES = [

]

setup(
    name='mrf-reconstruction-midl',
    version='0.1.0',
    description='On the Spatial and Temporal Influence for the Reconstruction of Magnetic Resonance Fingerprinting',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Fabian Balsiger',
    author_email='fabian.balsiger@artorg.unibe.ch',
    url='https://openreview.net/forum?id=HyeuSq9ke4',
    license='GNU GENERAL PUBLIC LICENSE',
    packages=find_packages(exclude=['test', 'docs']),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=[
        'image reconstruction',
        'deep learning',
        'magnetic resonance fingerprinting'
    ]
)
