from setuptools import setup, find_packages
setup(name='dxl-learn',
      version='0.0.4',
      description='Machine learn library.',
      url='https://github.com/Hong-Xiang/dxlearn',
      author='Hong Xiang',
      author_email='hx.hongxiang@gmail.com',
      license='MIT',
      packages=['dxl.learn'],
      package_dir={'': 'src/python'},
      install_requires=['dxl-fs', 'click'],
      zip_safe=False)
