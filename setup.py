
from setuptools import find_packages
from setuptools import setup

exec(open('ziln_cltv/version.py').read())

setup(
    name='ziln_cltv',
    version = __version__,
    description='A python package to build customer lifetime value(CLTV) models using ZILN loss & neural networks',
    author='DJ',
    author_email='willofdeepak@gmail.com',
    license='',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
                        'numpy >= 1.11.1',
                        'pandas',
                        'sklearn',
                        'lifetime_value',
                        'tensorflow',
                        'scipy']
)