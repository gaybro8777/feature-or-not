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

# Set up and preprocess dataset
pull_requests_dataframe = pd.read_csv("/pull_requests.csv")

pull_requests_dataframe = pull_requests_dataframe.reindex(
        np.random.permutation(pull_requests_dataframe.index))

pull_requests_dataframe["body_char_count"] = (
    pull_requests_dataframe["body"].str.len())

relevant_features = ['additions_count', 'deletions_count', 'changed_files_count', 'body_char_count', 'time_to_review_in_minutes']

# Filter out rows where relevant features are null
pull_requests_dataframe = pull_requests_dataframe.dropna(subset=(['cycle_time_in_days'] + relevant_features))

# Trim outliers
pull_requests_dataframe["additions_count"] = (
    pull_requests_dataframe["additions_count"].apply(lambda x: min(x, 10000)))

pull_requests_dataframe["deletions_count"] = (
    pull_requests_dataframe["deletions_count"].apply(lambda x: min(x, 10000)))

pull_requests_dataframe["changed_files_count"] = (
    pull_requests_dataframe["changed_files_count"].apply(lambda x: min(x, 300)))

pull_requests_dataframe["time_to_review_in_minutes"] = (
    pull_requests_dataframe["time_to_review_in_minutes"].apply(lambda x: min(x, 1000)))

# There is a huge amount of variability here - cap at 5 days
pull_requests_dataframe["cycle_time_in_days"] = (
    pull_requests_dataframe["cycle_time_in_days"].apply(lambda x: min(x, 5)))

print(pull_requests_dataframe.describe())

"""
We will first predict time-to-review based on the size of the PR

See: https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises
"""

def preprocess_features(pull_requests_dataframe):
  """Prepares input features from California housing data set.

  Args:
    pull_requests_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = pull_requests_dataframe[
      relevant_features
    ]
  processed_features = selected_features.copy()

  return processed_features

def preprocess_targets(pull_requests_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    pull_requests_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in minutes
  output_targets["cycle_time_in_minutes"] = (
      pull_requests_dataframe["cycle_time_in_days"] * 24 * 60)
  return output_targets

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.
  
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
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["cycle_time_in_minutes"], 
      batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["cycle_time_in_minutes"], 
      num_epochs=1, 
      shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
      validation_examples, validation_targets["cycle_time_in_minutes"], 
      num_epochs=1, 
      shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Create a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(training_predictions)
  calibration_data["targets"] = pd.Series(training_targets["cycle_time_in_minutes"])
  print(calibration_data.describe())

  # Plot predicted vs. actual
  plt.title("Predictions vs. Actual")
  plt.xlabel("predicted_cycle_time_in_minutes")
  plt.ylabel("actual_cycle_time_in_minutes")
  plt.scatter(calibration_data["predictions"], calibration_data["targets"])
  plt.savefig("/tmp/fig.png")

  # Output model bias + weights
  print([(v, linear_regressor.get_variable_value(v)) for v in linear_regressor.get_variable_names()])

  return linear_regressor

"""
End helper fns
"""
# Set up training + validation examples
training_examples = preprocess_features(pull_requests_dataframe.head(2000))
print(training_examples.describe())

training_targets = preprocess_targets(pull_requests_dataframe.head(2000))
print(training_targets.describe())

validation_examples = preprocess_features(pull_requests_dataframe.tail(500))
print(validation_examples.describe())

validation_targets = preprocess_targets(pull_requests_dataframe.tail(500))
print(validation_targets.describe())

linear_regressor = train_model(
    learning_rate=0.00001,
    steps=5000,
    batch_size=200,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
