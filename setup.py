# setup.py file for the pip package.

from setuptools import setup, find_packages
from os import path
from neuralnetwork_foundations.version import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='neuralnetwork_foundations',
    version=__version__,
    description='A repository for learning the foundations of neural networks and deep learning',
    long_description=long_description,
    url='https://github.com/krmayankb/NeuralNetwork-Foundations.git', 
    author='Mayank Kumar',
    project_urls={
        'Source': 'https://github.com/krmayankb/NeuralNetwork-Foundations.git',
    },  

    # Classifiers help users find your project by categorizing it.
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology", 
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        # We believe this package works will these versions, but we do not guarantee it!
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python',
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Build Tools",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.7", # 
    keywords="machine_learning deep_learning neural_networks neural_network_foundations computer_vision algorithms data_science data_analysis data_visualization data_engineering natural_language_processing nlp cv ml dl regression basic_neural_networks",
    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
        'torch',
        'torchvision',
        'tqdm',
        "opencv-python",
    ],
)

