"""Build of the C kernels (the only part of the packaging not in pyproject.toml).

The kernels (``src/anemoi_datasets_codecs/icon_lattice/csrc``) are compiled into one extension module,
``anemoi_datasets_codecs.icon_lattice._native``, which native.py loads with ctypes:

* ``variant.c`` is compiled once per ISA variant into its own object: ``scalar`` (``-DLC_SCALAR``: simd.h emulates
  the AVX2 intrinsics in portable C; every platform) and, on x86-64, ``avx2`` (``-mavx2``). A variant is a unity
  build of rans.c + decode.c + encode.c whose functions are all static, so the two link side by side;
* ``dispatch.c`` exports the entry points and picks a variant at load time (cpuid; ``$ANEMOI_DATASETS_CODECS_ISA``).

Flags: ``-O3 -fopenmp -ffp-contract=off``. ``-ffp-contract=off`` forbids fused multiply-add anywhere, so the
dequantisation ``offset + step * (float)n`` is two IEEE float32 roundings and the recipe / mask formulas round
exactly like numpy's float64: the encoder (numpy) verifies the bound on exactly the values the decoder produces, and
every build (either variant, any compiler honouring the flag) decodes bit-identically. The AVX2 variant targets the
plain AVX2 baseline (no FMA, no ``-march=native``), so one binary runs on every x86-64-v3 CPU, and the scalar
variant on anything else. The module uses only the stable ABI (abi3 wheels for CPython >= 3.11).

``$CC`` selects the compiler (default: ``gcc`` when on PATH, else Python's). With gcc, the directory of the
compiler's libgomp is added as an rpath when it is not a system directory, so the module loads without
``LD_LIBRARY_PATH`` (wheels for distribution would vendor libgomp with auditwheel instead).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError

CSRC = "src/anemoi_datasets_codecs/icon_lattice/csrc"
X86 = platform.machine().lower() in ("x86_64", "amd64")
COMMON = [
    "-O3",
    "-fopenmp",
    "-ffp-contract=off",
    "-fPIC",
    "-fvisibility=hidden",
    "-std=gnu11",
    "-Wall",
    "-Wno-unused-parameter",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-Wno-psabi",  # the scalar variant passes 32-byte generic vectors between always-inline functions
]
VARIANTS = [("scalar", ["-DLC_SCALAR"])] + ([("avx2", ["-mavx2"])] if X86 else [])
HEADERS = [str(p) for p in Path(CSRC).glob("*.[ch]")]
NEEDS = "anemoi-datasets-codecs compiles its C kernels at install time and needs a C compiler with OpenMP (gcc)"

if "CC" not in os.environ and shutil.which("gcc"):
    os.environ["CC"] = "gcc"


def _gomp_rpath() -> list[str]:
    cc = os.environ.get("CC", "cc").split()[0]
    try:
        out = subprocess.run([cc, "-print-file-name=libgomp.so.1"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    path = Path(out.stdout.strip())
    if not path.is_absolute() or not path.exists():
        return []
    d = path.resolve().parent
    if str(d) in ("/lib", "/lib64", "/usr/lib", "/usr/lib64") or str(d).startswith("/usr/lib/"):
        return []
    return [f"-Wl,-rpath,{d}"]


class BuildExt(build_ext):
    """Compile the ISA variants into separate objects, then link them with dispatch.c."""

    def build_extension(self, ext: Extension) -> None:
        cc = (getattr(self.compiler, "compiler_so", None) or ["cc"])[0]
        if shutil.which(cc) is None:
            raise SystemExit(f"{NEEDS}: the compiler {cc!r} was not found (set CC to one)")
        objects = []
        for name, flags in VARIANTS:
            try:
                objects += self.compiler.compile(
                    [f"{CSRC}/variant.c"],
                    output_dir=os.path.join(self.build_temp, name),
                    macros=[("LC_VARIANT", name)],
                    include_dirs=[CSRC],
                    extra_postargs=COMMON + flags,
                    depends=HEADERS,
                )
            except CompileError as e:
                raise SystemExit(f"{NEEDS}: compiling the {name} kernels with {cc!r} failed ({e})") from None
        ext.extra_objects = objects
        ext.extra_link_args = ["-fopenmp", *_gomp_rpath()]
        super().build_extension(ext)


setup(
    ext_modules=[
        Extension(
            "anemoi_datasets_codecs.icon_lattice._native",
            sources=[f"{CSRC}/dispatch.c"],
            include_dirs=[CSRC],
            define_macros=[("Py_LIMITED_API", "0x030B0000")] + ([("LC_HAVE_AVX2", "1")] if X86 else []),
            extra_compile_args=COMMON,
            libraries=["m"],
            depends=HEADERS,
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExt},
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
