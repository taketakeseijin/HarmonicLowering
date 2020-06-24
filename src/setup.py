from setuptools import setup, Extension
from torch.utils import cpp_extension

module_name = "interp_same"
cpp_files = ["./csrc/interp_same_size.cpp","./csrc/interp_same_size_kernel.cu"]
#inc = ["/usr/include"]
setup(name=module_name,
      ext_modules=[Extension(
            name=module_name,
            sources=cpp_files,
            include_dirs=cpp_extension.include_paths(),
            language='c++')],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

