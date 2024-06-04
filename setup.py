from setuptools import setup, find_packages

setup(
    name='agentlace',
    version='0.1.3',
    packages=find_packages(),
    description='library to enable distributed agent for ml training and inference',
    url='https://github.com/youliangtan/agentlace',
    author='auth',
    author_email='tan_you_liang@hotmail.com',
    license='MIT',
    install_requires=[
        'zmq',
        'typing',
        'typing_extensions',
        'opencv-python',
        'lz4',
        'gym>=0.26.0',
    ],
    zip_safe=False
)
