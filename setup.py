from setuptools import setup, find_packages

setup(
    name='nsfp',
    version='0.1.0',
    packages=['nsfp'], # Explicitly define the package
    package_dir={'nsfp': '.'}, # Tell setuptools that 'nsfp' package content is in the current directory
    install_requires=[
        # It's recommended to install PyTorch and Open3D separately
        # as they often require specific CUDA versions or build configurations.
        # e.g., conda install pytorch torchvision cudatoolkit -c pytorch
        # e.g., conda install -c open3d-admin open3d
    ],
    author='na-trium-144', # Replace with actual author
    description='Neural Scene Flow Prior (NeurIPS 2021 spotlight)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/na-trium-144/Neural_Scene_Flow_Prior', # Replace with actual URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Assuming MIT from LICENSE file
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7', # Based on conda create -n sf python=3.7
)
