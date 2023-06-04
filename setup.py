from setuptools import setup, find_packages

setup(
    name='fractionalDNN',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Center-for-Math-AI/Fractional_DNNs.git',
    license='MIT',
    author='Harbir Antil and Deepanshu Verma',
    author_email='hantil@gmu.edu',
    description='pytorch fDNN',
    install_requires=['torch', 'torchvision', 'matplotlib', 'numpy', 'pandas']
)
