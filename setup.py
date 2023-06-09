from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
VERSION = '0.3.0'
NAME = 'lmppl'
LICENSE = 'MIT License'
setup(
    name=NAME,
    packages=find_packages(exclude=['tests', 'misc', 'asset']),
    version=VERSION,
    license=LICENSE,
    description='Calculate perplexity on the text with pre-trained language models.',
    url='https://github.com/asahi417/lmppl',
    download_url="https://github.com/asahi417/lmppl/archive/v{}.tar.gz".format(VERSION),
    keywords=['language model', 't5', 'gpt3', 'bert', 'perplexity', 'nlp'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        f'License :: OSI Approved :: {LICENSE}',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "torch",
        "tqdm",
        "requests",
        "transformers",
        "sentencepiece",
        "accelerate",
        "openai",
        "protobuf<3.20"  # required by DeBERTa models
    ],
    python_requires='>=3.6',
    # entry_points={
    #     'console_scripts': [
    #         # 'lmqg-qae = lmqg.lmqg_cl.model_evaluation_qa_based_metric:main_qa_model_training'
    #     ]
    # }
)

