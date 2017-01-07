"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


__all_candidate_sampler_outputs = ["sampled_candidates",
                                  "true_expected_count",
                                  "sampled_expected_count"]


_AllCandidateSamplerOutput = collections.namedtuple("AllCandidateSampler",
                                                    __all_candidate_sampler_outputs)


def _all_candidate_sampler(true_classes, num_true, num_sampled, unique,
                           seed=None, seed2=None, name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to produce per batch.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).
    sampled_candidates: A `Tensor` of type `int64`. A vector of length num_sampled, in which each element is
      the ID of a sampled candidate.
    true_expected_count: A `Tensor` of type `float32`. A batch_size * num_true matrix, representing
      the number of times each candidate is expected to occur in a batch
      of sampled candidates. If unique=true, then this is a probability.
    sampled_expected_count: A `Tensor` of type `float32`. A vector of length num_sampled, for each sampled
      candidate representing the number of times the candidate is expected
      to occur in a batch of sampled candidates.  If unique=true, then this is a
      probability.
  """
  result = _op_def_lib.apply_op("AllCandidateSampler",
                                true_classes=true_classes, num_true=num_true,
                                num_sampled=num_sampled, unique=unique,
                                seed=seed, seed2=seed2, name=name)
  return _AllCandidateSamplerOutput._make(result)


__compute_accidental_hits_outputs = ["indices", "ids", "weights"]


_ComputeAccidentalHitsOutput = collections.namedtuple("ComputeAccidentalHits",
                                                      __compute_accidental_hits_outputs)


def _compute_accidental_hits(true_classes, sampled_candidates, num_true,
                             seed=None, seed2=None, name=None):
  r"""Computes the ids of the positions in sampled_candidates that match true_labels.

  When doing log-odds NCE, the result of this op should be passed through a
  SparseToDense op, then added to the logits of the sampled candidates. This has
  the effect of 'removing' the sampled labels that match the true labels by
  making the classifier sure that they are sampled labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      The true_classes output of UnpackSparseLabels.
    sampled_candidates: A `Tensor` of type `int64`.
      The sampled_candidates output of CandidateSampler.
    num_true: An `int`. Number of true labels per context.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, ids, weights).
    indices: A `Tensor` of type `int32`. A vector of indices corresponding to rows of true_candidates.
    ids: A `Tensor` of type `int64`. A vector of IDs of positions in sampled_candidates that match a true_label
      for the row with the corresponding index in indices.
    weights: A `Tensor` of type `float32`. A vector of the same length as indices and ids, in which each element
      is -FLOAT_MAX.
  """
  result = _op_def_lib.apply_op("ComputeAccidentalHits",
                                true_classes=true_classes,
                                sampled_candidates=sampled_candidates,
                                num_true=num_true, seed=seed, seed2=seed2,
                                name=name)
  return _ComputeAccidentalHitsOutput._make(result)


__fixed_unigram_candidate_sampler_outputs = ["sampled_candidates",
                                            "true_expected_count",
                                            "sampled_expected_count"]


_FixedUnigramCandidateSamplerOutput = collections.namedtuple("FixedUnigramCandidateSampler",
                                                             __fixed_unigram_candidate_sampler_outputs)


def _fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled,
                                     unique, range_max, vocab_file=None,
                                     distortion=None, num_reserved_ids=None,
                                     num_shards=None, shard=None,
                                     unigrams=None, seed=None, seed2=None,
                                     name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  A unigram sampler could use a fixed unigram distribution read from a
  file or passed in as an in-memory array instead of building up the distribution
  from data on the fly. There is also an option to skew the distribution by
  applying a distortion power to the weights.

  The vocabulary file should be in CSV-like format, with the last field
  being the weight associated with the word.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample per batch.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    vocab_file: An optional `string`. Defaults to `""`.
      Each valid line in this file (which should have a CSV-like format)
      corresponds to a valid word ID. IDs are in sequential order, starting from
      num_reserved_ids. The last entry in each line is expected to be a value
      corresponding to the count or relative probability. Exactly one of vocab_file
      and unigrams needs to be passed to this op.
    distortion: An optional `float`. Defaults to `1`.
      The distortion is used to skew the unigram probability distribution.
      Each weight is first raised to the distortion's power before adding to the
      internal unigram distribution. As a result, distortion = 1.0 gives regular
      unigram sampling (as defined by the vocab file), and distortion = 0.0 gives
      a uniform distribution.
    num_reserved_ids: An optional `int`. Defaults to `0`.
      Optionally some reserved IDs can be added in the range [0,
      ..., num_reserved_ids) by the users. One use case is that a special unknown
      word token is used as ID 0. These IDs will have a sampling probability of 0.
    num_shards: An optional `int` that is `>= 1`. Defaults to `1`.
      A sampler can be used to sample from a subset of the original range
      in order to speed up the whole computation through parallelism. This parameter
      (together with 'shard') indicates the number of partitions that are being
      used in the overall computation.
    shard: An optional `int` that is `>= 0`. Defaults to `0`.
      A sampler can be used to sample from a subset of the original range
      in order to speed up the whole computation through parallelism. This parameter
      (together with 'num_shards') indicates the particular partition number of a
      sampler op, when partitioning is being used.
    unigrams: An optional list of `floats`. Defaults to `[]`.
      A list of unigram counts or probabilities, one per ID in sequential
      order. Exactly one of vocab_file and unigrams should be passed to this op.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).
    sampled_candidates: A `Tensor` of type `int64`. A vector of length num_sampled, in which each element is
      the ID of a sampled candidate.
    true_expected_count: A `Tensor` of type `float32`. A batch_size * num_true matrix, representing
      the number of times each candidate is expected to occur in a batch
      of sampled candidates. If unique=true, then this is a probability.
    sampled_expected_count: A `Tensor` of type `float32`. A vector of length num_sampled, for each sampled
      candidate representing the number of times the candidate is expected
      to occur in a batch of sampled candidates.  If unique=true, then this is a
      probability.
  """
  result = _op_def_lib.apply_op("FixedUnigramCandidateSampler",
                                true_classes=true_classes, num_true=num_true,
                                num_sampled=num_sampled, unique=unique,
                                range_max=range_max, vocab_file=vocab_file,
                                distortion=distortion,
                                num_reserved_ids=num_reserved_ids,
                                num_shards=num_shards, shard=shard,
                                unigrams=unigrams, seed=seed, seed2=seed2,
                                name=name)
  return _FixedUnigramCandidateSamplerOutput._make(result)


__learned_unigram_candidate_sampler_outputs = ["sampled_candidates",
                                              "true_expected_count",
                                              "sampled_expected_count"]


_LearnedUnigramCandidateSamplerOutput = collections.namedtuple("LearnedUnigramCandidateSampler",
                                                               __learned_unigram_candidate_sampler_outputs)


def _learned_unigram_candidate_sampler(true_classes, num_true, num_sampled,
                                       unique, range_max, seed=None,
                                       seed2=None, name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample per batch.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).
    sampled_candidates: A `Tensor` of type `int64`. A vector of length num_sampled, in which each element is
      the ID of a sampled candidate.
    true_expected_count: A `Tensor` of type `float32`. A batch_size * num_true matrix, representing
      the number of times each candidate is expected to occur in a batch
      of sampled candidates. If unique=true, then this is a probability.
    sampled_expected_count: A `Tensor` of type `float32`. A vector of length num_sampled, for each sampled
      candidate representing the number of times the candidate is expected
      to occur in a batch of sampled candidates.  If unique=true, then this is a
      probability.
  """
  result = _op_def_lib.apply_op("LearnedUnigramCandidateSampler",
                                true_classes=true_classes, num_true=num_true,
                                num_sampled=num_sampled, unique=unique,
                                range_max=range_max, seed=seed, seed2=seed2,
                                name=name)
  return _LearnedUnigramCandidateSamplerOutput._make(result)


__log_uniform_candidate_sampler_outputs = ["sampled_candidates",
                                          "true_expected_count",
                                          "sampled_expected_count"]


_LogUniformCandidateSamplerOutput = collections.namedtuple("LogUniformCandidateSampler",
                                                           __log_uniform_candidate_sampler_outputs)


def _log_uniform_candidate_sampler(true_classes, num_true, num_sampled,
                                   unique, range_max, seed=None, seed2=None,
                                   name=None):
  r"""Generates labels for candidate sampling with a log-uniform distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample per batch.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).
    sampled_candidates: A `Tensor` of type `int64`. A vector of length num_sampled, in which each element is
      the ID of a sampled candidate.
    true_expected_count: A `Tensor` of type `float32`. A batch_size * num_true matrix, representing
      the number of times each candidate is expected to occur in a batch
      of sampled candidates. If unique=true, then this is a probability.
    sampled_expected_count: A `Tensor` of type `float32`. A vector of length num_sampled, for each sampled
      candidate representing the number of times the candidate is expected
      to occur in a batch of sampled candidates.  If unique=true, then this is a
      probability.
  """
  result = _op_def_lib.apply_op("LogUniformCandidateSampler",
                                true_classes=true_classes, num_true=num_true,
                                num_sampled=num_sampled, unique=unique,
                                range_max=range_max, seed=seed, seed2=seed2,
                                name=name)
  return _LogUniformCandidateSamplerOutput._make(result)


__thread_unsafe_unigram_candidate_sampler_outputs = ["sampled_candidates",
                                                    "true_expected_count",
                                                    "sampled_expected_count"]


_ThreadUnsafeUnigramCandidateSamplerOutput = collections.namedtuple("ThreadUnsafeUnigramCandidateSampler",
                                                                    __thread_unsafe_unigram_candidate_sampler_outputs)


def _thread_unsafe_unigram_candidate_sampler(true_classes, num_true,
                                             num_sampled, unique, range_max,
                                             seed=None, seed2=None,
                                             name=None):
  r"""Generates labels for candidate sampling with a learned unigram distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample per batch.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).
    sampled_candidates: A `Tensor` of type `int64`. A vector of length num_sampled, in which each element is
      the ID of a sampled candidate.
    true_expected_count: A `Tensor` of type `float32`. A batch_size * num_true matrix, representing
      the number of times each candidate is expected to occur in a batch
      of sampled candidates. If unique=true, then this is a probability.
    sampled_expected_count: A `Tensor` of type `float32`. A vector of length num_sampled, for each sampled
      candidate representing the number of times the candidate is expected
      to occur in a batch of sampled candidates.  If unique=true, then this is a
      probability.
  """
  result = _op_def_lib.apply_op("ThreadUnsafeUnigramCandidateSampler",
                                true_classes=true_classes, num_true=num_true,
                                num_sampled=num_sampled, unique=unique,
                                range_max=range_max, seed=seed, seed2=seed2,
                                name=name)
  return _ThreadUnsafeUnigramCandidateSamplerOutput._make(result)


__uniform_candidate_sampler_outputs = ["sampled_candidates",
                                      "true_expected_count",
                                      "sampled_expected_count"]


_UniformCandidateSamplerOutput = collections.namedtuple("UniformCandidateSampler",
                                                        __uniform_candidate_sampler_outputs)


def _uniform_candidate_sampler(true_classes, num_true, num_sampled, unique,
                               range_max, seed=None, seed2=None, name=None):
  r"""Generates labels for candidate sampling with a uniform distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample per batch.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).
    sampled_candidates: A `Tensor` of type `int64`. A vector of length num_sampled, in which each element is
      the ID of a sampled candidate.
    true_expected_count: A `Tensor` of type `float32`. A batch_size * num_true matrix, representing
      the number of times each candidate is expected to occur in a batch
      of sampled candidates. If unique=true, then this is a probability.
    sampled_expected_count: A `Tensor` of type `float32`. A vector of length num_sampled, for each sampled
      candidate representing the number of times the candidate is expected
      to occur in a batch of sampled candidates.  If unique=true, then this is a
      probability.
  """
  result = _op_def_lib.apply_op("UniformCandidateSampler",
                                true_classes=true_classes, num_true=num_true,
                                num_sampled=num_sampled, unique=unique,
                                range_max=range_max, seed=seed, seed2=seed2,
                                name=name)
  return _UniformCandidateSamplerOutput._make(result)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AllCandidateSampler"
  input_arg {
    name: "true_classes"
    type: DT_INT64
  }
  output_arg {
    name: "sampled_candidates"
    type: DT_INT64
  }
  output_arg {
    name: "true_expected_count"
    type: DT_FLOAT
  }
  output_arg {
    name: "sampled_expected_count"
    type: DT_FLOAT
  }
  attr {
    name: "num_true"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "num_sampled"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "unique"
    type: "bool"
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "ComputeAccidentalHits"
  input_arg {
    name: "true_classes"
    type: DT_INT64
  }
  input_arg {
    name: "sampled_candidates"
    type: DT_INT64
  }
  output_arg {
    name: "indices"
    type: DT_INT32
  }
  output_arg {
    name: "ids"
    type: DT_INT64
  }
  output_arg {
    name: "weights"
    type: DT_FLOAT
  }
  attr {
    name: "num_true"
    type: "int"
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "FixedUnigramCandidateSampler"
  input_arg {
    name: "true_classes"
    type: DT_INT64
  }
  output_arg {
    name: "sampled_candidates"
    type: DT_INT64
  }
  output_arg {
    name: "true_expected_count"
    type: DT_FLOAT
  }
  output_arg {
    name: "sampled_expected_count"
    type: DT_FLOAT
  }
  attr {
    name: "num_true"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "num_sampled"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "unique"
    type: "bool"
  }
  attr {
    name: "range_max"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "vocab_file"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "distortion"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "num_reserved_ids"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "num_shards"
    type: "int"
    default_value {
      i: 1
    }
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shard"
    type: "int"
    default_value {
      i: 0
    }
    has_minimum: true
  }
  attr {
    name: "unigrams"
    type: "list(float)"
    default_value {
      list {
      }
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "LearnedUnigramCandidateSampler"
  input_arg {
    name: "true_classes"
    type: DT_INT64
  }
  output_arg {
    name: "sampled_candidates"
    type: DT_INT64
  }
  output_arg {
    name: "true_expected_count"
    type: DT_FLOAT
  }
  output_arg {
    name: "sampled_expected_count"
    type: DT_FLOAT
  }
  attr {
    name: "num_true"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "num_sampled"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "unique"
    type: "bool"
  }
  attr {
    name: "range_max"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "LogUniformCandidateSampler"
  input_arg {
    name: "true_classes"
    type: DT_INT64
  }
  output_arg {
    name: "sampled_candidates"
    type: DT_INT64
  }
  output_arg {
    name: "true_expected_count"
    type: DT_FLOAT
  }
  output_arg {
    name: "sampled_expected_count"
    type: DT_FLOAT
  }
  attr {
    name: "num_true"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "num_sampled"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "unique"
    type: "bool"
  }
  attr {
    name: "range_max"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "ThreadUnsafeUnigramCandidateSampler"
  input_arg {
    name: "true_classes"
    type: DT_INT64
  }
  output_arg {
    name: "sampled_candidates"
    type: DT_INT64
  }
  output_arg {
    name: "true_expected_count"
    type: DT_FLOAT
  }
  output_arg {
    name: "sampled_expected_count"
    type: DT_FLOAT
  }
  attr {
    name: "num_true"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "num_sampled"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "unique"
    type: "bool"
  }
  attr {
    name: "range_max"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "UniformCandidateSampler"
  input_arg {
    name: "true_classes"
    type: DT_INT64
  }
  output_arg {
    name: "sampled_candidates"
    type: DT_INT64
  }
  output_arg {
    name: "true_expected_count"
    type: DT_FLOAT
  }
  output_arg {
    name: "sampled_expected_count"
    type: DT_FLOAT
  }
  attr {
    name: "num_true"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "num_sampled"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "unique"
    type: "bool"
  }
  attr {
    name: "range_max"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
