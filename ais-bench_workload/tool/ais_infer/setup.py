from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
setup(
    name='ais_infer',
    version='0.0.2',
    description='ais-bench inference tool',
    long_description=long_description,
    url='https://gitee.com/ascend/tools/ais-bench_workload/tool/tools/ais_infer',
    packages=find_packages(exclude='ais_infer.py'),
    keywords='ais-bench inference tool',
    install_requires=required,
    python_requires='>=3.7'
)