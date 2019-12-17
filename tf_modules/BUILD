# This package contains custom TensorFlow modules to be used across valan.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "lingunet4_3x18_1x18",
    srcs = ["lingunet4_3x18_1x18.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:utils",
    ],
)

pytype_strict_library(
    name = "lingunet3_1x8_1x18",
    srcs = ["lingunet3_1x8_1x18.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:utils",
    ],
)
