# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
#include "PokerProperties.hpp"
#include "Player.hpp"
#include "GameEngine.hpp"
ext_modules = [
    Pybind11Extension(
        "poker_engine",
        ["bindings.cpp", "poker.cpp", "Player.cpp", "GameEngine.cpp"],
        cxx_std=20,
        define_macros=[("POKER_PYBIND", "1")],
    )
]

setup(
    name="poker_engine",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
