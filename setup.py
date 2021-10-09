
import numpy
import subprocess
from setuptools import setup, find_packages
#from setuptools.command.develop import develop
from distutils.core import setup, Extension
                          
udp_ext = Extension('cosmic.camera.udpframereader',
                           sources = ['cosmic/camera/udpframereader.c'],
                           extra_compile_args=['-std=c99','-march=native','-O3'],
                           extra_link_args=['-std=c99','-march=native'])
print(find_packages())
setup(
    name='cosmic',
    version='0.0.1',
    description='A selection of ptycho scripts developed for ptychography Beamlines at the ALS',
    long_description="",
    author='Bjoern Enders, David Shapiro and others',
    include_package_data=True,
    ext_modules=[udp_ext],
    requires=['numpy (>=1.8.2)', 'setuptools', 'tifffile', 'msgpack', 'msgpack_numpy'],
    license="ALS private use for now",
    zip_safe=False,
    #packages=['cosmic',
    #          'cosmic.ext',
    #          'cosmic.utils',
    #          'cosmic.io',
    #          'cosmic.preprocess']	
    
    packages = find_packages(),
)
