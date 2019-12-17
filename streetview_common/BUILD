# This package contains code shared between all StreetView-based environments.
# https://arxiv.org/abs/1811.12354

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "run_metadata_writer",
    srcs = ["run_metadata_writer.py"],
    deps = [
        ":streetview_constants",
        "//tensorflow:tensorflow_py",
        "@com_google_absl//absl/flags:flag",
    ],
)

py_library(
    name = "streetview_env",
    srcs = ["streetview_env.py"],
    deps = [
        ":graph_loader",
        ":streetview_constants",
        "//third_party/py/language/bert:tokenization",
        "//third_party/py/valan/framework:base_env",
        "//third_party/py/valan/framework:common",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "example_env_config",
    srcs = ["example_env_config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/valan/framework:hparam",
    ],
)

pytype_strict_library(
    name = "streetview_constants",
    srcs = ["streetview_constants.py"],
    deps = [],
)

py_library(
    name = "baseline_agent",
    srcs = ["baseline_agent.py"],
    deps = [
        ":streetview_constants",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:base_agent",
        "//third_party/py/valan/framework:common",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "graph_loader",
    srcs = ["graph_loader.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "env_test_base",
    srcs = ["env_test_base.py"],
    deps = [
        ":streetview_constants",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:utils",
    ],
)
