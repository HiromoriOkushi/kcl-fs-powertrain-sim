from setuptools import setup, find_packages

setup(
    name='kcl_fs_powertrain',
    version='0.1.0',
    description='Formula Student Powertrain Simulation Digital Twin',
    author='Hiromori Okushi',
    author_email='', # Add email if desired
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'pyyaml>=6.0',
        'CoolProp>=6.4.0',
        # 'plotly>=5.3.0', 
        'pytest>=6.2.0',
        'shapely>=1.8.0',
        'gpxpy>=1.5.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Choose appropriate license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.8',
)