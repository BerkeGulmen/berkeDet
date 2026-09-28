import sys

from setuptools import Extension, setup

extra = ["/O2"] if sys.platform == "win32" else ["-O3"]

setup(
    ext_modules=[
        Extension(
            "berkeDet._core",
            sources=["src/berkeDet/_core.c"],
            extra_compile_args=extra,
        )
    ]
)
