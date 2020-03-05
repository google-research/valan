# This package contains code for training navigation agent in Matterport3D
# environment using multitask learning and/or environment-agnostic learning.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = [
        "//third_party/py/valan/google/r2r:__subpackages__",
    ],
)

pytype_strict_library(
    name = "mt_agent_config",
    srcs = ["mt_agent_config.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/py/valan/framework:hparam"],
)

pytype_strict_library(
    name = "mt_env_config",
    srcs = ["mt_env_config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/valan/framework:hparam",
        "//third_party/py/valan/r2r:constants",
        "//third_party/py/valan/r2r:env_config",
        "//third_party/py/valan/r2r:env_ndh_config",
    ],
)

pytype_strict_library(
    name = "mt_problem",
    srcs = ["mt_problem.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mt_agent",
        ":mt_agent_config",
        ":mt_env",
        "//tensorflow:tensorflow_py",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:problem_type",
        "//third_party/py/valan/r2r:constants",
    ],
)

pytype_strict_library(
    name = "mt_env",
    srcs = ["mt_env.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mt_env_config",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/r2r:constants",
        "//third_party/py/valan/r2r:env",
        "//third_party/py/valan/r2r:env_ndh",
    ],
)

pytype_strict_library(
    name = "mt_agent",
    srcs = ["mt_agent.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:base_agent",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/r2r:constants",
        "//third_party/py/valan/r2r:image_encoder",
        "//third_party/py/valan/r2r:instruction_encoder",
    ],
)
