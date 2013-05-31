from setuptools import setup, find_packages

setup(name="facerec",
  version=0.1,
  download_url='https://github.com/jayrambhia/facerec/zipball',
  author='Jay Rambhia',
  author_email='jayrambhia777@gmail.com',
  license='BSD',
  packages = find_packages(),
  requires=['cv2', 'numpy']
  )
