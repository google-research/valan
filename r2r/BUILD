# Code for R2R based on Matterport3D environment.

load("//devtools/python/blaze:python3.bzl", "py2and3_strict_test")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

licenses(["notice"])

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = [
        "//research/lux/earthsea:__subpackages__",
    ],
)

pytype_strict_library(
    name = "instruction_encoder",
    srcs = ["instruction_encoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_library(
    name = "r2r_problem",
    srcs = ["r2r_problem.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":agent",
        ":agent_config",
        ":constants",
        ":curriculum_env",
        ":curriculum_env_config",
        ":env",
        ":env_config",
        ":eval_metric",
        "//tensorflow:tensorflow_py",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:problem_type",
    ],
)

pytype_strict_library(
    name = "env",
    srcs = ["env.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":env_config",
        ":house_parser",
        ":image_features_py_pb2",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:base_env",
        "//third_party/py/valan/framework:common",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "house_utils",
    srcs = ["house_utils.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/py/networkx"],
)

pytype_strict_library(
    name = "agent_config",
    srcs = ["agent_config.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/py/valan/framework:hparam"],
)

pytype_strict_library(
    name = "env_config",
    srcs = ["env_config.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        "//third_party/py/valan/framework:eval_metric",
        "//third_party/py/valan/framework:hparam",
    ],
)

pytype_strict_library(
    name = "ndh_problem",
    srcs = ["ndh_problem.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":agent",
        ":agent_config",
        ":constants",
        ":env_ndh",
        ":env_ndh_config",
        ":eval_metric",
        "//tensorflow:tensorflow_py",
        "//third_party/py/tensorflow_probability",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:problem_type",
    ],
)

pytype_strict_library(
    name = "image_encoder",
    srcs = ["image_encoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_library(
    name = "eval_metric",
    srcs = ["eval_metric.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        "//tensorflow:tensorflow_py",
        "//third_party/py/matplotlib",
        "//third_party/py/networkx",
        "//third_party/py/valan/framework:eval_metric",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_library(
    name = "env_ndh_config",
    srcs = ["env_ndh_config.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":env_config",
        "//third_party/py/valan/framework:hparam",
    ],
)

pytype_strict_library(
    name = "curriculum_env_config",
    srcs = ["curriculum_env_config.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":env_config",
        "//third_party/py/valan/framework:hparam",
    ],
)

pytype_strict_library(
    name = "agent",
    srcs = ["agent.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":image_encoder",
        ":instruction_encoder",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:base_agent",
        "//third_party/py/valan/framework:common",
    ],
)

pytype_strict_library(
    name = "house_parser",
    srcs = ["house_parser.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":house_utils",
        "//tensorflow:tensorflow_py",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY2AND3",
)

pytype_strict_library(
    name = "curriculum_env",
    srcs = ["curriculum_env.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":curriculum_env_config",
        ":env",
    ],
)

pytype_strict_library(
    name = "env_ndh",
    srcs = ["env_ndh.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":env",
        ":env_ndh_config",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:common",
    ],
)

pytype_strict_library(
    name = "env_test_lib",
    srcs = ["env_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":constants",
        ":env",
        ":env_config",
        ":image_features_py_pb2",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:hparam",
        "@absl_py//absl/flags",
    ],
)

proto_library(
    name = "image_features_proto",
    srcs = ["image_features.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "image_features_py_pb2",
    api_version = 2,
    deps = [":image_features_proto"],
)

cc_proto_library(
    name = "image_features_cc_proto",
    deps = [":image_features_proto"],
)

filegroup(
    name = "test_data",
    srcs = glob(["testdata/image_features/*"]) + [
        "testdata/R2R_small_split.json",
        "testdata/vocab.txt",
        "testdata/connections/gZ6f7yhEvPG_connectivity.json",
        "testdata/connections/p5wJjkQkbXX_connectivity.json",
        "testdata/scans/p5wJjkQkbXX/house_segmentations/p5wJjkQkbXX.house",
        "testdata/scans/gZ6f7yhEvPG/house_segmentations/gZ6f7yhEvPG.house",
        "testdata/NDH/small_split.json",
        "testdata/NDH/vocab.txt",
    ],
)

py2and3_strict_test(
    name = "agent_test",
    srcs = ["agent_test.py"],
    data = [":test_data"],
    deps = [
        ":agent",
        ":agent_config",
        ":constants",
        ":env",
        ":env_config",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:hparam",
        "//third_party/py/valan/framework:utils",
        "@absl_py//absl/flags",
    ],
)

py2and3_strict_test(
    name = "curriculum_env_test",
    srcs = ["curriculum_env_test.py"],
    data = [":test_data"],
    deps = [
        ":constants",
        ":curriculum_env",
        ":curriculum_env_config",
        ":env_test_lib",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
    ],
)

py2and3_strict_test(
    name = "image_encoder_test",
    srcs = ["image_encoder_test.py"],
    deps = [
        ":image_encoder",
        "//tensorflow:tensorflow_py",
    ],
)

py2and3_strict_test(
    name = "instruction_encoder_test",
    srcs = ["instruction_encoder_test.py"],
    deps = [
        ":instruction_encoder",
        "//tensorflow:tensorflow_py",
    ],
)

py2and3_strict_test(
    name = "env_test",
    srcs = ["env_test.py"],
    data = [":test_data"],
    deps = [
        ":constants",
        ":env",
        ":env_config",
        ":image_features_py_pb2",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:hparam",
        "@absl_py//absl/flags",
    ],
)

py2and3_strict_test(
    name = "eval_metric_test",
    srcs = ["eval_metric_test.py"],
    data = [":test_data"],
    deps = [
        ":constants",
        ":env",
        ":env_config",
        ":eval_metric",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:hparam",
        "@absl_py//absl/flags",
    ],
)

py2and3_strict_test(
    name = "env_ndh_test",
    srcs = ["env_ndh_test.py"],
    data = [":test_data"],
    deps = [
        ":constants",
        ":env_ndh",
        ":env_ndh_config",
        ":env_test_lib",
        "//tensorflow:tensorflow_py",
        "//third_party/py/valan/framework:common",
        "//third_party/py/valan/framework:hparam",
        "@absl_py//absl/flags",
    ],
)

py2and3_strict_test(
    name = "house_parser_test",
    srcs = ["house_parser_test.py"],
    data = [":test_data"],
    deps = [
        ":house_parser",
        ":house_utils",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
    ],
)
