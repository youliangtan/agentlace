from setuptools import setup

setup(
    name='edgeml',
    version='0.1',
    description='library to enable distributed edge ml training and inference',
    url='https://github.com/youliangtan/edgeml',
    author='auth',
    author_email='tan_you_liang@hotmail.com',
    license='MIT',
    install_requires=['zmq', 'typing'],
    zip_safe=False
)
