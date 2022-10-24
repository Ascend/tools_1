import os
from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

def prepare_packet_file():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    cmd = "cp {}/ais_infer.py {}/__main__.py;".format(cur_path, cur_path)
    print("run cmd:{}".format(cmd))
    os.system(cmd)

print("lcm debug xxxx")
# prepare_packet_file()
print("lcm debug xxxx")

setup(
    name='ais_infer',
    version='0.1.0',
    description='ais-bench inference tool',
    long_description=long_description,
    url='https://gitee.com/ascend/tools/ais-bench_workload/tool/tools/ais_infer',
    packages=find_packages(),
    keywords='ais-bench inference tool',
    install_requires=required,
    python_requires='>=3.7'
)