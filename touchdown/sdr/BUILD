load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//learning/brain/contrib/learn/tpu:tpu.bzl", "cloud_tpu_py_binaries")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "lingunet",
    srcs = ["lingunet.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:utils",
    ],
)

cloud_tpu_py_binaries(
    name = "train_eager",
    srcs = [
        "train_eager.py",
    ],
    main = "train_eager.py",
    deps = [
        ":lingunet",
        "//tensorflow:tensorflow_py",
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@six_archive//:six",
    ],
)
