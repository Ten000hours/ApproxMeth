from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='fastrsqrt_cpp',						# 编译后的链接库名称
    ext_modules=[
        cpp_extension.CppExtension(
            'fastrsqrt_cpp', ['fastrsqrt.cpp']		       # 待编译文件，及编译函数
        )
    ],
    cmdclass={						       # 执行编译命令设置
        'build_ext': cpp_extension.BuildExtension
    }
)