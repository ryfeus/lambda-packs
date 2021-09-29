from sklearn.datasets import make_blobs
from sklearn import svm
from time import perf_counter
import numpy as np

def handler(event, context):
  num_of_cycles = event.get('num_of_cycles', 10)
  list_train_data = []
  list_test_data = []
  list_train_timings = []
  list_prediction_timings = []

  for i in range(num_of_cycles):
    list_train_data.append(make_blobs(
      n_samples=event.get('n_samples', 10000),
      centers=3,
      n_features=event.get('n_features', 1024)
    ))
    list_test_data.append(make_blobs(
      n_samples=event.get('n_test_samples', 10000),
      centers=3,
      n_features=event.get('n_features', 1024)
    ))

  for i in range(num_of_cycles):
    X_train, y_train = list_train_data[i]
    X_test, y_test = list_test_data[i]

    t0 = perf_counter()
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    list_train_timings.append(perf_counter() - t0)

    t0 = perf_counter()
    clf.predict(X_test)
    list_prediction_timings.append(perf_counter() - t0)

  print("Mean train", np.mean(list_train_timings))
  print("Std train", np.std(list_train_timings))
  print("Mean inference", np.mean(list_prediction_timings))
  print("Std inference", np.std(list_prediction_timings))

  return {
    'Mean train': np.mean(list_train_timings),
    'Std train': np.std(list_train_timings),
    'Mean inference': np.mean(list_prediction_timings),
    'Std inference': np.std(list_prediction_timings)
  }

