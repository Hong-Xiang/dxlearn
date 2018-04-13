from setuptools import setup, find_packages
setup(
    name='dxl-learn',
    version='0.0.6',
    description='Machine learn library.',
    url='https://github.com/Hong-Xiang/dxlearn',
    author='Hong Xiang',
    author_email='hx.hongxiang@gmail.com',
    license='MIT',
    packages=[
        'dxl.learn',
        'dxl.learn.core',
    ],
    package_dir={'': 'src/python'},
    install_requires=['dxl-fs', 'click', 'dxl-shape'],
    zip_safe=False)
