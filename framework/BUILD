# VALAN: Vision and Language Agent Navigation
# Framework libraries and tests

licenses(["notice"])

package(default_visibility = ["//research/lux/earthsea:__subpackages__"])

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
    deps = ["@absl_py//absl/flags"],
)

py_library(
    name = "actor",
    srcs = ["actor.py"],
    deps = [
        ":actor_config",
        ":common",
        ":learner_config",
        ":problem_type",
        ":utils",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@seed_rl//:grpc",
        "@six_archive//:six",
    ],
)

py_library(
    name = "learner",
    srcs = ["learner.py"],
    deps = [
        ":actor_config",
        ":base_agent",
        ":common",
        ":learner_config",
        ":loss_fns",
        ":problem_type",
        ":utils",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/rnn:rnn_py",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@seed_rl//:grpc",
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
        ":learner_config",
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
        "@absl_py//absl/flags",
        "@seed_rl//:grpc",
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
        "@absl_py//absl/flags",
        "@seed_rl//:grpc",
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
        ":actor_config",
        ":common",
        ":learner_config",
        ":problem_type",
        ":utils",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@seed_rl//:grpc",
    ],
)

py_test(
    name = "eval_actor_test",
    srcs = ["eval_actor_test.py"],
    python_version = "PY3",
    deps = [
        ":eval_actor",
        ":testing_utils",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
        "@seed_rl//:grpc",
    ],
)

py_library(
    name = "eval_aggregator",
    srcs = ["eval_aggregator.py"],
    deps = [
        ":actor_config",
        ":common",
        ":learner_config",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@seed_rl//:grpc",
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
        "@absl_py//absl/flags",
        "@seed_rl//:grpc",
    ],
)
