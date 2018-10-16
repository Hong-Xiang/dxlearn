from setuptools import setup, find_packages
setup(
    name='dxl-learn',
    version='0.2.1',
    description='Machine learn library.',
    url='https://github.com/tech-pi/dxlearn',
    author='Hong Xiang',
    author_email='hx.hongxiang@gmail.com',
    license='MIT',
    namespace_packages=['dxl'],
    packages=find_packages('src/python'),
    package_dir={'': 'src/python'},
    install_requires=[
        'tables',
        'scipy',
        'typing',
        'arrow',
        'h5py',
        'click',
        'tensorflow',
        'fs',
        'pathlib',
        'numpy',
        'tqdm',
        'doufo==0.0.4',
        'dxl-core==0.1.7'
    ],
    entry_points="""
        [console_scripts]
        learn=dxl.learn.cli.main:dxlearn
    """,
    zip_safe=False)
