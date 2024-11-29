from setuptools import setup, find_packages

setup(
    name='rl_lab',
    version='0.1',
    packages=find_packages(),
    url='http://example.com/my_package',
    license='MIT',
    author='hsh',
    author_email='1653996628@qq.com',
    description='None',
    install_requires=[
        "psutil",
        "lxml",
        "transformations",
        "pybullet",
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.1.0",
    ],
    python_requires='>=3.10',
)