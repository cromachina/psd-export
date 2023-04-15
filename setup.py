from setuptools import setup, find_packages

setup(
    name='psd-export',
    version='1.0.1',
    author='cromachina',
    description='Fast exporting of PSDs with [tagged] layers for variants.',
    long_description=(
        open('README.md').read()
    ),
    long_description_content_type='text/markdown',
    url='https://github.com/cromachina/psd-export',
    license='MIT',
    install_requires=[
        'numpy',
        'opencv_python',
        'psd_tools',
        'psutil',
        'pyrsistent',
    ],
    package_dir={'':'src'},
    packages=find_packages('src'),
    entry_points={'console_scripts': ['psd-export=psd_export.export:main']},
    keywords='exporter psd art',
    classifiers=[
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    ],
)