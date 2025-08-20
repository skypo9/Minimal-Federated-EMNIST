"""
tff_emnist_local.py

Federated Learning on EMNIST using TensorFlow Federated (TFF)
----------------------------------------------------------------
This script demonstrates a simple federated learning workflow on a subset of the EMNIST dataset.
You can configure the number of training rounds and clients per round via command-line arguments.

Main steps:
1. Load and preprocess EMNIST data for federated learning.
2. Define a simple Keras CNN model for digit classification.
3. Wrap the model for TFF and set up federated averaging and evaluation.
4. Run federated training for the specified number of rounds.
5. Evaluate and log metrics for each round, saving results to a CSV file.

Usage example:
    python tff_emnist_local.py --rounds 5 --clients-per-round 5
"""
# -----------------------
# 1) Imports and argument parsing
# -----------------------
# -----------------------
# 2) Data loading and preprocessing
# -----------------------
# -----------------------
# 3) Model definition
# -----------------------
# -----------------------
# 4) Federated learning setup (legacy TFF API)
# -----------------------
# -----------------------
# 5) Training loop and metrics logging
# -----------------------
# Suppress TensorFlow and asyncio warnings for cleaner output
import os
import warnings
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=RuntimeWarning)
import argparse
import random
import csv
import tensorflow as tf
import tensorflow_federated as tff

def build_datasets(only_digits: bool, train_pool_size: int, test_pool_size: int,
                   batch_size: int, epochs_per_client: int, shuffle_buffer: int):
    # Load federated EMNIST (grouped by writer IDs -> perfect for FL)
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=only_digits)

    train_pool = sorted(emnist_train.client_ids)[:train_pool_size]
    test_pool  = sorted(emnist_test.client_ids)[:test_pool_size]

    def preprocess(ds):
        def _map(ex):
            x = tf.expand_dims(tf.cast(ex['pixels'], tf.float32) / 255.0, -1)  # [28,28]->[28,28,1]
            y = tf.cast(ex['label'], tf.int32)
            return x, y
        return (ds.map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                  .shuffle(shuffle_buffer)
                  .batch(batch_size)
                  .repeat(epochs_per_client))

    def make_federated(client_ids, clientdata):
        return [preprocess(clientdata.create_tf_dataset_for_client(cid)) for cid in client_ids]

    # Create a dummy dataset to extract input_spec
    sample_ds = make_federated([train_pool[0]], emnist_train)[0]
    input_spec = sample_ds.element_spec

    return (emnist_train, emnist_test, train_pool, test_pool, input_spec)

def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

def build_algorithms(input_spec):
    def model_fn():
        return tff.learning.from_keras_model(
            create_keras_model(),
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    fedavg = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    )
    fed_eval = tff.learning.build_federated_evaluation(model_fn)
    return fedavg, fed_eval

def main():
    p = argparse.ArgumentParser(description="TFF FedAvg on EMNIST (local simulation)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--train-pool", type=int, default=20)
    p.add_argument("--test-pool", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--epochs-per-client", type=int, default=1)
    p.add_argument("--shuffle-buffer", type=int, default=100)
    p.add_argument("--only-digits", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="metrics.csv")
    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Build data & specs
    emnist_train, emnist_test, train_pool, test_pool, input_spec = build_datasets(
        only_digits=args.only_digits,
        train_pool_size=args.train_pool,
        test_pool_size=args.test_pool,
        batch_size=args.batch_size,
        epochs_per_client=args.epochs_per_client,
        shuffle_buffer=args.shuffle_buffer,
    )

    def make_federated(client_ids, clientdata):
        # Re-define here to close over args (batch-size/epochs/etc.)
        def preprocess(ds):
            def _map(ex):
                x = tf.expand_dims(tf.cast(ex['pixels'], tf.float32) / 255.0, -1)
                y = tf.cast(ex['label'], tf.int32)
                return x, y
            return (ds.map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                      .shuffle(args.shuffle_buffer)
                      .batch(args.batch_size)
                      .repeat(args.epochs_per_client))
        return [preprocess(clientdata.create_tf_dataset_for_client(cid)) for cid in client_ids]

    # Fixed test shard (small, deterministic)
    fixed_test_ids = test_pool[:5]
    federated_test = make_federated(fixed_test_ids, emnist_test)

    # Build algorithms
    fedavg, fed_eval = build_algorithms(input_spec)
    state = fedavg.initialize()
    # No eval_state for legacy API

    # CSV logger
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round","train_loss","train_acc","eval_loss","eval_acc"])
        writer.writeheader()

        for r in range(1, args.rounds + 1):
            round_clients = random.sample(train_pool, args.clients_per_round)
            federated_train = make_federated(round_clients, emnist_train)

            state, train_metrics = fedavg.next(state, federated_train)


            # Run evaluation with legacy API
            eval_metrics = fed_eval(state.model, federated_test)

            row = {
                "round": r,
                "train_loss": float(train_metrics["train"]["loss"]),
                "train_acc": float(train_metrics["train"]["sparse_categorical_accuracy"]),
                "eval_loss": float(eval_metrics["loss"]),
                "eval_acc": float(eval_metrics["sparse_categorical_accuracy"]),
            }
            writer.writerow(row)
            print(f"Round {r:02d}  train_acc={row['train_acc']:.3f}  eval_acc={row['eval_acc']:.3f}")

    print(f"\nSaved metrics to {args.out}")

if __name__ == "__main__":
    main()
