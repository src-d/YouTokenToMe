import io
import os

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = cythonize([
    Extension(
        "_youtokentome_cython",
        [
            "youtokentome/cpp/yttm.pyx",
            "youtokentome/cpp/bpe.cpp",
            "youtokentome/cpp/utils.cpp",
            "youtokentome/cpp/utf8.cpp",
        ],
        extra_compile_args=["-std=c++14", "-pthread", "-O3", "-static", "-flto"],
        language="c++",
    )
])

with io.open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    LONG_DESCRIPTION = "\n" + f.read()

setup(
    name="youtokentome-srcd",
    version="2.0.0",
    packages=find_packages(),
    description="Unsupervised text tokenizer focused on computational efficiency",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/src-d/youtokentome",
    python_requires=">=3.5.0",
    install_requires=["Click>=7.0"],
    entry_points={"console_scripts": ["yttm = youtokentome.yttm_cli:main"]},
    author="Ivan Belonogov; source{d}",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Cython",
        "Programming Language :: C++",
    ],
    package_data={"": [
        "youtokentome/cpp/utils.h",
        "youtokentome/cpp/bpe.h",
        "youtokentome/cpp/utf8.h",
        "youtokentome/cpp/third_party/flat_hash_map.h",
        "youtokentome/cpp/third_party/LICENSE",
        "LICENSE",
        "README.md",
        "requirements.txt",
    ],},
    ext_modules=extensions,
)
