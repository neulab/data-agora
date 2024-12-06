from setuptools import setup, find_packages

setup(
    name='llama-recipes',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.2',
        'accelerate',
        'appdirs',
        'loralib',
        'bitsandbytes',
        'black',
        'black[jupyter]',
        'datasets',
        'fire',
        'peft',
        'transformers>=4.43.1',
        'sentencepiece',
        'py7zr',
        'scipy',
        'optimum',
        'matplotlib',
        'gradio',
        'chardet',
        'openai',
        'typing-extensions==4.8.0',
        'tabulate',
        'evaluate',
        'rouge_score',
        'pyyaml==6.0.1',
        'faiss-gpu; python_version < \'3.11\'',
        'unstructured[pdf]',
        'sentence_transformers',
        'codeshield',
    ]
)


