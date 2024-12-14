from setuptools import setup, find_packages

setup(name='unitree_mujoco',
      version='1.0.1',
      author='UnitreeRobotics',
      author_email='unitree@unitree.com',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Unitree robot sdk version 2 for python',
      project_urls={
            "Source Code": "https://github.com/unitreerobotics/unitree_sdk2_python",
      },
      python_requires='>=3.8',
      install_requires=[
            "cyclonedds==0.10.2",
            "numpy",
            "opencv-python",
            
      ],
      )