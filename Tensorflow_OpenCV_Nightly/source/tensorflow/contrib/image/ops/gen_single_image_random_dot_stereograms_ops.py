"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections as _collections

from google.protobuf import text_format as _text_format

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2

# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library

def single_image_random_dot_stereograms(depth_values,
                                        hidden_surface_removal=None,
                                        convergence_dots_size=None,
                                        dots_per_inch=None,
                                        eye_separation=None, mu=None,
                                        normalize=None, normalize_max=None,
                                        normalize_min=None, border_level=None,
                                        number_colors=None,
                                        output_image_shape=None,
                                        output_data_window=None, name=None):
  r"""Outputs a single image random dot stereogram for export via encode_PNG/JPG OP.

  Given the 2-D tensor 'depth_values' with encoded Z values, this operation will
  encode 3-D data into a 2-D image.  The output of this Op is suitable for the
  encode_PNG/JPG ops.  Be careful with image compression as this may corrupt the
  encode 3-D data witin the image.

  This Op is based upon:
  'http://www.learningace.com/doc/4331582/b6ab058d1e206d68ab60e4e1ead2fe6e/sirds-paper'

  Example use which outputs a SIRDS image as picture_out.png:
  ```python
  img=[[1,2,3,3,2,1],
       [1,2,3,4,5,2],
       [1,2,3,4,5,3],
       [1,2,3,4,5,4],
       [6,5,4,4,5,5]]

  session = tf.InteractiveSession()

  sirds = single_image_random_dot_stereograms(img,convergence_dots_size=8,number_colors=256,normalize=True)

  out = sirds.eval()

  png = tf.image.encode_png(out).eval()

  with open('picture_out.png', 'wb') as f:
      f.write(png)
  ```

  Args:
    depth_values: A `Tensor`. Must be one of the following types: `float64`, `float32`, `int64`, `int32`.
      Z values of data to encode into 'output_data_window' window,
      lower values are further away {0.0 floor(far), 1.0 ceiling(near) after normalization}, must be 2-D tensor
    hidden_surface_removal: An optional `bool`. Defaults to `True`.
      Activate hidden surface removal
    convergence_dots_size: An optional `int`. Defaults to `8`.
      Black dot size in pixels to help view converge image, drawn on bottom of image
    dots_per_inch: An optional `int`. Defaults to `72`.
      Output device in dots/inch
    eye_separation: An optional `float`. Defaults to `2.5`.
      Separation between eyes in inches
    mu: An optional `float`. Defaults to `0.3333`.
      Depth of field, Fraction of viewing distance (eg. 1/3 = .3333)
    normalize: An optional `bool`. Defaults to `True`.
      Normalize input data to [0.0, 1.0]
    normalize_max: An optional `float`. Defaults to `-100`.
      Fix MAX value for Normalization - if < MIN, autoscale
    normalize_min: An optional `float`. Defaults to `100`.
      Fix MIN value for Normalization - if > MAX, autoscale
    border_level: An optional `float`. Defaults to `0`.
      Value of border depth 0.0 {far} to 1.0 {near}
    number_colors: An optional `int`. Defaults to `256`.
      2 (Black & White),256 (grayscale), and Numbers > 256 (Full Color) are all that are supported currently
    output_image_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[1024, 768, 1]`.
      Output size of returned image in X,Y, Channels 1-grayscale, 3 color (1024, 768, 1),
      channels will be updated to 3 if 'number_colors' > 256
    output_data_window: An optional `tf.TensorShape` or list of `ints`. Defaults to `[1022, 757]`.
      Size of "DATA" window, must be equal to or smaller than 'output_image_shape', will be centered
      and use 'convergence_dots_size' for best fit to avoid overlap if possible
    name: A name for the operation (optional).

  Returns:
    A tensor of size 'output_image_shape' with the encloded 'depth_values'
  """
  result = _op_def_lib.apply_op("SingleImageRandomDotStereograms",
                                depth_values=depth_values,
                                hidden_surface_removal=hidden_surface_removal,
                                convergence_dots_size=convergence_dots_size,
                                dots_per_inch=dots_per_inch,
                                eye_separation=eye_separation, mu=mu,
                                normalize=normalize,
                                normalize_max=normalize_max,
                                normalize_min=normalize_min,
                                border_level=border_level,
                                number_colors=number_colors,
                                output_image_shape=output_image_shape,
                                output_data_window=output_data_window,
                                name=name)
  return result


_ops.RegisterShape("SingleImageRandomDotStereograms")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "SingleImageRandomDotStereograms"
  input_arg {
    name: "depth_values"
    type_attr: "T"
  }
  output_arg {
    name: "image"
    type: DT_UINT8
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
        type: DT_INT64
        type: DT_INT32
      }
    }
  }
  attr {
    name: "hidden_surface_removal"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "convergence_dots_size"
    type: "int"
    default_value {
      i: 8
    }
  }
  attr {
    name: "dots_per_inch"
    type: "int"
    default_value {
      i: 72
    }
  }
  attr {
    name: "eye_separation"
    type: "float"
    default_value {
      f: 2.5
    }
  }
  attr {
    name: "mu"
    type: "float"
    default_value {
      f: 0.3333
    }
  }
  attr {
    name: "normalize"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "normalize_max"
    type: "float"
    default_value {
      f: -100
    }
  }
  attr {
    name: "normalize_min"
    type: "float"
    default_value {
      f: 100
    }
  }
  attr {
    name: "border_level"
    type: "float"
    default_value {
      f: 0
    }
  }
  attr {
    name: "number_colors"
    type: "int"
    default_value {
      i: 256
    }
  }
  attr {
    name: "output_image_shape"
    type: "shape"
    default_value {
      shape {
        dim {
          size: 1024
        }
        dim {
          size: 768
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    name: "output_data_window"
    type: "shape"
    default_value {
      shape {
        dim {
          size: 1022
        }
        dim {
          size: 757
        }
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
