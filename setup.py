from setuptools import setup, find_packages
setup(
    name='dxl-learn',
    version='0.2.0',
    description='Machine learn library.',
    url='https://github.com/tech-pi/dxlearn',
    author='Hong Xiang',
    author_email='hx.hongxiang@gmail.com',
    license='MIT',
    namespace_packages=['dxl'],
    packages=find_packages('src/python'),
    package_dir={'': 'src/python'},
    install_requires=['jfs', 'click', 'dxl-shape>=0.1.1',
                      'dxl-core>=0.1.6',
                      'doufo>=0.0.3'
                      'arrow', 'tqdm'],
    entry_points="""
        [console_scripts]
        learn=dxl.learn.cli.main:dxlearn
    """,
    zip_safe=False)
