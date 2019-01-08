from setuptools import setup

setup(name='pytet',
      version='0.1.1',
      description='Type Extension Trees Python implementation',
      url='',
      author='Giovanni Pellegrini',
      author_email='giovanni.pellegrini@unitn.it',
      license='MIT',
      packages=['pytet'],
      install_requires=[
          'autograd',
          'scikit-learn',
          'anytree',
      ],
      zip_safe=False)

