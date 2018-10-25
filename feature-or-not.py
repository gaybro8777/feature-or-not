#!/usr/bin/env python3

import math

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

pull_requests_dataframe = pd.read_csv("/pull_requests.csv")

pull_requests_dataframe = pull_requests_dataframe.reindex(
        np.random.permutation(pull_requests_dataframe.index))

# Filter out rows where relevant features are null
pull_requests_dataframe = pull_requests_dataframe.dropna(subset=['size', 'cycle_time_in_days'])

"""
To get started with ML, first predict one label from one feature
We will predict time-to-review based on the size of the PR

See: https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises
"""
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(learning_rate, steps, batch_size, input_feature):
  """Trains a linear regression model.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `pull_requests_dataframe`
      to use as input feature.
      
  Returns:
    A Pandas `DataFrame` containing targets and the corresponding predictions done
    after training the model.
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = pull_requests_dataframe[[my_feature]].astype('float32')
  my_label = "cycle_time_in_minutes"
  targets = pull_requests_dataframe[my_label].astype('float32')

  # Create input functions.
  training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create feature columns.
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
    
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state model's line.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = pull_requests_dataframe.sample(n=500)
  plt.scatter(sample[my_feature], sample[my_label])

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
      metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)

  print("Model training finished.")

  # Plot sampled data vs. prediction line
  y_extents = np.array([0, sample[my_label].max()])

  weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
  bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

  x_extents = (y_extents - bias) / weight
  x_extents = np.maximum(np.minimum(x_extents,
    sample[my_feature].max()),
    sample[my_feature].min())
  y_extents = weight * x_extents + bias
  plt.plot(x_extents, y_extents) 

  # Create a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  print(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
  print([(v, linear_regressor.get_variable_value(v)) for v in linear_regressor.get_variable_names()])

  # Plot predicted vs. actual
  plt.subplot(1, 2, 2)
  plt.title("Predictions vs. Actual")
  plt.xlabel("predicted_cycle_time_in_minutes")
  plt.ylabel("actual_cycle_time_in_minutes")
  plt.scatter(calibration_data["predictions"], calibration_data["targets"])
  plt.savefig("/tmp/fig.png")

  # Output model bias + weights
  print(linear_regressor.model_fn)

  return calibration_data

pull_requests_dataframe["size"] = (
    pull_requests_dataframe["size"]).apply(lambda x: min(x, 10000))

pull_requests_dataframe["cycle_time_in_days"] = (
    pull_requests_dataframe["cycle_time_in_days"]).apply(lambda x: min(x, 20))

pull_requests_dataframe["cycle_time_in_minutes"] = (
    pull_requests_dataframe["cycle_time_in_days"]).apply(lambda x: x * 24 * 60)

calibration_data = train_model(
    learning_rate=0.00001,
    steps=5000,
    batch_size=200,
    input_feature="size")
