import setuptools
setuptools.setup(name='speaker_separation',
                 version='1.0.0',
                 description='DPRNN model for speaker separation',
                 author='esb18849',
                 author_email='RHChen-18849@email.esunbank.com.tw',
                 license='MIT',
                 install_requires=['torch==1.8.1', 'speechbrain==0.5.4', 'scipy==1.5.2', 'numpy==1.19.1', 'torchaudio==0.8.1'],
                 include_package_data=True,
                 packages=setuptools.find_packages('src'),
                 package_dir={'':'src'}
                )