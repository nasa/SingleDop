from setuptools import setup

setup(
    name='SingleDop',
    version='0.5',
    author='Timothy Lang',
    author_email='timothy.j.lang@nasa.gov',
    packages=['singledop',],
    #url='http://pypi.python.org/pypi/PyDisdrometer/',
    license='LICENSE.txt',
    description='Single Doppler Retrieval Toolkit',
    long_description=open('description.txt').read(),
    install_requires=['arm_pyart', 'pytda','xray'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console"
        ],

)
