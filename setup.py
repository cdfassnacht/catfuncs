import setuptools

# with open('README.md', 'r', encoding='utf-8') as fh:
#     long_description = fh.read()

setuptools.setup(
    name='catfuncs',
    version='1.0',
    author='Chris Fassnacht',
    author_email='cdfassnacht@ucdavis.edu',
    description='Code for astronomical catalogs',
    # long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cdfassnacht/catfuncs',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.10',
        'scipy>=1.1',
        'astropy>=3.1',
        'matplotlib>=3.0',
        'cdfutils',
        'specim'
    ],
)
