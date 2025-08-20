# Federated Learning on EMNIST with TensorFlow Federated

This project demonstrates federated learning using the EMNIST dataset and TensorFlow Federated (TFF).

## File: `tff_emnist_local.py`

This script runs a federated learning experiment on a subset of the EMNIST dataset using TFF. It allows you to specify the number of training rounds and the number of clients per round.

---

## How to Use

### 1. **Environment Setup**
- Use Python 3.8 or 3.9 for best compatibility.
- Create and activate a virtual environment:
  ```bash
  python -m venv .venv
  # On Windows:
  .venv\Scripts\activate
  # On Mac/Linux:
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install tensorflow tensorflow_federated
  ```

### 2. **Run the Script**
- Basic usage:
  ```bash
  python tff_emnist_local.py --rounds 5 --clients-per-round 5
  ```
- Arguments:
  - `--rounds`: Number of federated training rounds (default: 5)
  - `--clients-per-round`: Number of clients sampled per round (default: 5)
  - `--out`: Output CSV file for metrics (default: metrics.csv)

### 3. **Output**
- The script prints training and evaluation accuracy for each round.
- Metrics are saved to a CSV file (default: `metrics.csv`) with columns:
  - `round`, `train_loss`, `train_acc`, `eval_loss`, `eval_acc`

### 4. **Analyzing Results**
- Open the CSV file in Excel or any spreadsheet tool to visualize or analyze the results.
- Each row corresponds to a training round.

---

## Script Steps Explained

1. **Data Loading:**
   - Loads a small subset of the EMNIST dataset using TFF's simulation API.
2. **Preprocessing:**
   - Prepares the data for Keras models (reshaping, batching, shuffling).
3. **Model Definition:**
   - Defines a simple CNN using Keras for digit classification.
4. **Federated Learning Setup:**
   - Uses TFF's legacy API to build the federated averaging process and evaluation function.
5. **Training Loop:**
   - For each round:
     - Randomly selects clients.
     - Trains the model on their data.
     - Evaluates the model on a fixed test set.
     - Logs and prints metrics.
6. **Saving Results:**
   - Metrics are written to a CSV file for later analysis.

---

## Notes
- If you see warnings about `set_wakeup_fd only works in main thread`, you can ignore them.
- For best results, use a compatible Python and TFF version as described above.
- You can modify the script to use more clients or rounds, but this will increase runtime.

---

## References
- [TensorFlow Federated Documentation](https://www.tensorflow.org/federated)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

Feel free to modify the script for your own federated learning experiments!
