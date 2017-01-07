"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_neg_train_outputs = [""]


def neg_train(w_in, w_out, examples, labels, lr, vocab_count,
              num_negative_samples, name=None):
  r"""Training via negative sampling.

  Args:
    w_in: A `Tensor` of type mutable `float32`. input word embedding.
    w_out: A `Tensor` of type mutable `float32`. output word embedding.
    examples: A `Tensor` of type `int32`. A vector of word ids.
    labels: A `Tensor` of type `int32`. A vector of word ids.
    lr: A `Tensor` of type `float32`.
    vocab_count: A list of `ints`. Count of words in the vocabulary.
    num_negative_samples: An `int`. Number of negative samples per example.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("NegTrain", w_in=w_in, w_out=w_out,
                                examples=examples, labels=labels, lr=lr,
                                vocab_count=vocab_count,
                                num_negative_samples=num_negative_samples,
                                name=name)
  return result


ops.RegisterShape("NegTrain")(None)
_skipgram_outputs = ["vocab_word", "vocab_freq", "words_per_epoch",
                    "current_epoch", "total_words_processed", "examples",
                    "labels"]


_SkipgramOutput = collections.namedtuple("Skipgram", _skipgram_outputs)


def skipgram(filename, batch_size, window_size=None, min_count=None,
             subsample=None, name=None):
  r"""Parses a text file and creates a batch of examples.

  Args:
    filename: A `string`. The corpus's text file name.
    batch_size: An `int`. The size of produced batch.
    window_size: An optional `int`. Defaults to `5`.
      The number of words to predict to the left and right of the target.
    min_count: An optional `int`. Defaults to `5`.
      The minimum number of word occurrences for it to be included in the
      vocabulary.
    subsample: An optional `float`. Defaults to `0.001`.
      Threshold for word occurrence. Words that appear with higher
      frequency will be randomly down-sampled. Set to 0 to disable.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (vocab_word, vocab_freq, words_per_epoch, current_epoch, total_words_processed, examples, labels).
    vocab_word: A `Tensor` of type `string`. A vector of words in the corpus.
    vocab_freq: A `Tensor` of type `int32`. Frequencies of words. Sorted in the non-ascending order.
    words_per_epoch: A `Tensor` of type `int64`. Number of words per epoch in the data file.
    current_epoch: A `Tensor` of type `int32`. The current epoch number.
    total_words_processed: A `Tensor` of type `int64`. The total number of words processed so far.
    examples: A `Tensor` of type `int32`. A vector of word ids.
    labels: A `Tensor` of type `int32`. A vector of word ids.
  """
  result = _op_def_lib.apply_op("Skipgram", filename=filename,
                                batch_size=batch_size,
                                window_size=window_size, min_count=min_count,
                                subsample=subsample, name=name)
  return _SkipgramOutput._make(result)


ops.RegisterShape("Skipgram")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "NegTrain"
  input_arg {
    name: "w_in"
    type: DT_FLOAT
    is_ref: true
  }
  input_arg {
    name: "w_out"
    type: DT_FLOAT
    is_ref: true
  }
  input_arg {
    name: "examples"
    type: DT_INT32
  }
  input_arg {
    name: "labels"
    type: DT_INT32
  }
  input_arg {
    name: "lr"
    type: DT_FLOAT
  }
  attr {
    name: "vocab_count"
    type: "list(int)"
  }
  attr {
    name: "num_negative_samples"
    type: "int"
  }
  is_stateful: true
}
op {
  name: "Skipgram"
  output_arg {
    name: "vocab_word"
    type: DT_STRING
  }
  output_arg {
    name: "vocab_freq"
    type: DT_INT32
  }
  output_arg {
    name: "words_per_epoch"
    type: DT_INT64
  }
  output_arg {
    name: "current_epoch"
    type: DT_INT32
  }
  output_arg {
    name: "total_words_processed"
    type: DT_INT64
  }
  output_arg {
    name: "examples"
    type: DT_INT32
  }
  output_arg {
    name: "labels"
    type: DT_INT32
  }
  attr {
    name: "filename"
    type: "string"
  }
  attr {
    name: "batch_size"
    type: "int"
  }
  attr {
    name: "window_size"
    type: "int"
    default_value {
      i: 5
    }
  }
  attr {
    name: "min_count"
    type: "int"
    default_value {
      i: 5
    }
  }
  attr {
    name: "subsample"
    type: "float"
    default_value {
      f: 0.001
    }
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
