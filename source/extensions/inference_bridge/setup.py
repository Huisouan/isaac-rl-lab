from setuptools import setup, find_packages

setup(
    name='unitree_bridge',  # 项目的名称
    version='0.1.0',  # 项目的版本号
    packages=find_packages(),  # 自动发现所有包和子包
    install_requires=[

    ],
    entry_points={
        'console_scripts': [
            # 如果你的项目包含命令行工具，可以在这里定义
            # 'command_name = package.module:function'
        ],
    },
    author='hsh',  # 作者的名字
    author_email='1653996628@qq.com',  # 作者的邮箱
    description='unitree_bridge',  # 项目的简短描述
    url='https://github.com/hsh/unitree_inference',  # 项目的主页
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 支持的Python版本
)