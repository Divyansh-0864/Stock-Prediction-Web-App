from setuptools import find_packages,setup
from typing import List
# every folder should have __init__.py file so that it is considered as package by setup.py


HYPEN_E_DOT='-e .' ## in requirements.txt to automatically trigger setup.py
def get_requirements(file_path:str)->List[str]:
    '''
    returns the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='Stock Price Predictor',
version='0.0.1',
author='Blackbll',
author_email='eniggma@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)