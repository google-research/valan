# VALAN: Vision and Language Agent Navigation
# Framework libraries and tests

load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
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

py_library(
    name = "base_env",
    srcs = ["base_env.py"],
    deps = [
        ":common",
        "@six_archive//:six",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
)

py_library(
    name = "actor",
    srcs = ["actor.py"],
    deps = [
        ":common",
        ":hyperparam_flags",
        ":problem_type",
        ":utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/grpc",
        "//third_party/py/typing",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@six_archive//:six",
    ],
)

py_library(
    name = "learner",
    srcs = ["learner.py"],
    deps = [
        ":base_agent",
        ":common",
        ":hyperparam_flags",
        ":log_ckpt_restoration",
        ":loss_fns",
        ":problem_type",
        ":utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "//tensorflow/cc/seed_rl/grpc",
        "//third_party/py/typing",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "base_agent",
    srcs = ["base_agent.py"],
    deps = [
        ":common",
        ":utils",
        "//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "problem_type",
    srcs = ["problem_type.py"],
    deps = ["@six_archive//:six"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//pyglib:gfile",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "vtrace",
    srcs = ["vtrace.py"],
    deps = ["//tensorflow:tensorflow_py"],
)

py_library(
    name = "loss_fns",
    srcs = ["loss_fns.py"],
    deps = [
        ":common",
        ":focal_loss",
        ":utils",
        ":vtrace",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "actor_test",
    srcs = ["actor_test.py"],
    python_version = "PY3",
    deps = [
        ":actor",
        ":actor_config",
        ":testing_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/grpc",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "learner_test",
    srcs = ["learner_test.py"],
    python_version = "PY3",
    deps = [
        ":learner",
        ":testing_utils",
        ":utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/grpc",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "base_agent_test",
    srcs = ["base_agent_test.py"],
    python_version = "PY3",
    deps = [
        ":base_agent",
        ":common",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    data = [
        ":test_data",
    ],
    python_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "vtrace_test",
    srcs = ["vtrace_test.py"],
    python_version = "PY3",
    deps = [
        ":vtrace",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "testing_utils",
    srcs = ["testing_utils.py"],
    deps = [
        ":base_agent",
        ":base_env",
        ":common",
        ":problem_type",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "actor_config",
    srcs = ["actor_config.py"],
    deps = ["@absl_py//absl/flags"],
)

py_library(
    name = "learner_config",
    srcs = ["learner_config.py"],
    deps = ["@absl_py//absl/flags"],
)

py_library(
    name = "eval_actor",
    srcs = ["eval_actor.py"],
    deps = [
        ":common",
        ":problem_type",
        ":utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/grpc",
        "//third_party/py/typing",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "eval_actor_test",
    srcs = ["eval_actor_test.py"],
    python_version = "PY3",
    deps = [
        ":eval_actor",
        ":hyperparam_flags",
        ":testing_utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/grpc",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "eval_aggregator",
    srcs = ["eval_aggregator.py"],
    deps = [
        ":common",
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/grpc",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@six_archive//:six",
    ],
)

py_test(
    name = "eval_aggregator_test",
    srcs = ["eval_aggregator_test.py"],
    python_version = "PY3",
    deps = [
        ":common",
        ":eval_aggregator",
        "//tensorflow:tensorflow_py",
        "//tensorflow/cc/seed_rl/grpc",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "eval_metric",
    srcs = ["eval_metric.py"],
)

py_library(
    name = "hparam",
    srcs = ["hparam.py"],
    deps = [
        "@six_archive//:six",
    ],
)

pytype_strict_library(
    name = "focal_loss",
    srcs = ["focal_loss.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

pytype_strict_library(
    name = "log_ckpt_restoration",
    srcs = ["log_ckpt_restoration.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "hyperparam_flags",
    srcs = ["hyperparam_flags.py"],
    srcs_version = "PY3",
    deps = ["@absl_py//absl/flags"],
)

py_strict_test(
    name = "focal_loss_test",
    srcs = ["focal_loss_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":focal_loss",
        "//tensorflow:tensorflow_py",
    ],
)

py_strict_test(
    name = "loss_fns_test",
    srcs = ["loss_fns_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":common",
        ":focal_loss",
        ":hyperparam_flags",
        ":loss_fns",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
    ],
)
