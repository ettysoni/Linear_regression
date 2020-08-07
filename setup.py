from setuptools import setup, find_packages, Command
from codecs import open
from os import path, system
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanCommand(Command):
    """Custom clean command to tidy up the project root.
    """
    
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        system('rm -vrf ./dist ./*.pyc ./*.tgz ./*.egg-info')

setup(

    name='Python Dependency Resolver',  # Required

    version='1.0.0',

    description='Github Projects',

    author='Etty Soni',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required

    install_requires = ['numpy',
                        'pandas',
                        'pylint',
                        'sklearn',
                        'seaborn'
                        ],

    cmdclass={
            'clean': CleanCommand
        }
)