import setuptools
setuptools.setup(name='speaker_separation',
                 version='1.0.2',
                 description='DPRNN model for speaker separation',
                 author='esb18849',
                 author_email='a824300@gmail.com',
                 license='MIT',
                 install_requires=['numpy>=1.19.1', 'scipy==1.5.4', 'torch>=1.8.1', 'speechbrain>=0.5.4', 'torchaudio>=0.8.1', 'wget>=3.2'],
                 include_package_data=True,
                 packages=setuptools.find_packages('src'),
                 package_dir={'': 'src'})
