import onnxruntime as rt
import numpy as np
from time import perf_counter
sess = None


def handler(event, context):
  global sess
  t0 = perf_counter()
  if sess is None:
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = event.get('num_of_threads', 2)
    if event.get('execution_mode', 'sequential') == 'sequential':
      sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    else:
      sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess = rt.InferenceSession("inception_v3.onnx")

  input_name = sess.get_inputs()[0].name
  output_name = sess.get_outputs()[0].name
  input_shape = sess.get_inputs()[0].shape

  print("Model initialized", perf_counter() - t0)

  list_imgs = []
  list_timings = []
  num_of_cycles = event.get('num_of_cycles', 10)
  
  for i in range(num_of_cycles):
    list_imgs.append(np.array(np.random.random_sample(input_shape), dtype=np.float32))

  for i in range(num_of_cycles):
    t0 = perf_counter()

    pred_onx = sess.run([output_name], {input_name: list_imgs[i]})[0]

    list_timings.append(perf_counter() - t0)

  print("Mean inference", np.mean(list_timings))
  print("Std inference", np.std(list_timings))

  return {'Mean inference': np.mean(list_timings), 'Std inference': np.std(list_timings)}

