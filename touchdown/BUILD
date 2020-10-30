# This package contains all the Valan code for problems involving Touchdown
# dataset.
# https://arxiv.org/abs/1811.12354
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# TODO(harshm): Try to change this back to pytype_strict_library.
py_library(
    name = "env",
    srcs = ["env.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/streetview_common:graph_loader",
        "//third_party/py/valan/streetview_common:streetview_constants",
        "//third_party/py/valan/streetview_common:streetview_env",
    ],
)

pytype_strict_library(
    name = "env_config",
    srcs = ["env_config.py"],
    deps = [
        "//third_party/py/valan/framework:hparam",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [
        "//third_party/py/valan/streetview_common:streetview_constants",
    ],
)

pytype_strict_library(
    name = "td_problem",
    srcs = ["td_problem.py"],
    deps = [
        ":constants",
        ":env",
        ":env_config",
        "//tensorflow:tensorflow_py",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:eval_metric",
        "//third_party/py/valan/framework:problem_type",
        "//third_party/py/valan/framework:utils",
        "//third_party/py/valan/streetview_common:baseline_agent",
        "//third_party/py/valan/streetview_common:run_metadata_writer",
        "//third_party/py/valan/streetview_common:streetview_constants",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "custom_flags",
    srcs = ["custom_flags.py"],
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)
