This repository implements a research experiment comparing several neural-network learning rules—not a normal image-classification application. Its goal is to test whether recurrent networks develop “brain-like” predictive responses, such as stronger activity when an expected image suddenly changes.

## Overall workflow

```text
MNIST or BAIR videos
        ↓
Generate 16-frame sequences
        ↓
Two-layer recurrent network
        ↓
Train with predictive, contrastive, supervised, or no learning
        ↓
Extract internal predictions and activations
        ↓
Compare correlations, mismatch responses, decodability, and gain control
```

### 1. Data preprocessing

Most preprocessing is in [preproc.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/preproc.py:24).

The code supports four experimental paradigms:

| Paradigm | What the network sees |
|---|---|
| Roving Oddball | MNIST digits repeated for several time steps, followed by an unexpected new digit |
| Omission | Similar sequences, but some images are replaced with blank stimuli |
| Statistical Learning | A digit moves according to a class-specific motion rule |
| Video Learning | Real 16-frame clips from the BAIR robot-pushing dataset |

For MNIST, images are normalized and converted into sequences of 16 frames. In the roving experiment, the same image is generally shown for 6–8 steps before another image appears. This creates “standard” and “oddball” events ([preproc.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/preproc.py:582)).

The statistical-learning condition gives each digit class a different motion pattern. Rotation and translation accumulate over time, producing predictable trajectories ([preproc.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/preproc.py:434)).

BAIR videos are reduced to 16 frames, resized to 28×28, normalized, and converted to grayscale ([preproc.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/preproc.py:99)).

### 2. Neural-network architecture

The model is created in [train_models.ipynb](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/train_models.ipynb).

Its main structure is:

```text
16 × 28 × 28 × 1 image sequence
        ↓ reshape
16 × 784
        ↓ normalization
SimpleRNN with 512 units
        ↓ normalization
SimpleRNN with 128 units
```

Both recurrent layers return an output at every time step. The activation is a leaky ReLU with slope `0.5`, for which the project also defines an inverse ([training.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/training.py:29)).

When recurrence is disabled, `FrozenSimpleRNN` replaces the recurrent matrix with a fixed zero matrix. That lets the authors compare recurrent temporal processing against a non-recurrent control ([training.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/training.py:12)).

### 3. The four training objectives

The same architecture can be trained four different ways.

- **Predictive:** The recurrent state is treated as a prediction of the following input. The code projects the recurrent prediction backwards through the network using pseudoinverses and minimizes its difference from the next image ([training.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/training.py:313)).

- **Contrastive:** A Forward-Forward-inspired loss calculates the “goodness” of a sequence as the sum of squared neural activity and tries to distinguish coherent sequences from shuffled ones ([training.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/training.py:341)).

- **Supervised:** A linear mapping is fitted from the latent activations to the provided targets, and the model is trained to make those targets linearly recoverable ([training.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/training.py:258)).

- **Untrained:** The initialized network is saved without optimization, serving as a baseline.

Training can also be:

- **Global:** Standard end-to-end `model.fit()`.
- **Local:** Each recurrent layer is trained separately, after which the dataset is transformed through that layer before training the next one ([training.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/training.py:188)).

Two regularization switches are examined:

- `control_gain`: L2 penalty on layer activity.
- `enforce_sparsity`: L2 penalty on network weights.

Despite the names, the “sparsity” option is ordinary L2 regularization rather than a direct sparsity constraint.

### 4. How predictions are extracted

The network does not have a decoder. Instead, the code mathematically reverses each layer:

1. Invert the activation.
2. Remove the bias.
3. Remove the recurrent contribution.
4. Multiply by the pseudoinverse of the input-weight matrix.
5. Continue backwards through earlier layers.

This reconstructs what the network’s recurrent state predicts at the input level ([training.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/training.py:70)).

Conceptually, for an RNN,

```text
h(t) = activation(x(t)·Winput + h(t-1)·Wrecurrent + bias)
```

the recurrent term

```text
h(t-1)·Wrecurrent
```

is interpreted as the network’s prior or prediction about the next input.

### 5. Analysis performed

[analysis.ipynb](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/analysis.ipynb) contains four principal analyses:

1. **Prior/input correlation**  
   It measures how strongly the recurrent prediction correlates with the next bottom-up signal. Fisher transformations are used for confidence intervals and comparisons ([analysis.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/analysis.py:24)).

2. **Mismatch response (MMR)**  
   A video either continues normally or switches identity halfway through. The code compares squared RNN activity between matching and mismatching sequences. This is intended to resemble a neural mismatch response ([analysis.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/analysis.py:50)).

3. **Information learned by the representation**  
   Ridge regression is fitted on the internal activations to test whether target variables can be decoded. The reported score is actually \(R^2\), even though some helper functions call it “accuracy.”

4. **Gain control**  
   Activation variance under low and high sensory noise is compared, with and without regularization. Wilcoxon tests examine whether regularized networks control their activation variance more effectively.

The plotting helpers in [visualization.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/visualization.py:11) display input sequences, reconstructed predictions, and average activity over time.

### 6. Saving results

Each model is saved using a condition identifier such as:

```text
exp_Video_Learning-algo_Predictive-loc_False-rec_True-gain_False-sparse_False
```

This is handled by [utils.py](C:/Users/khadidja/Desktop/PFE/predictive_coding_algorithms/src/utils.py:6).

## Current state and important limitations

The scientific pipeline has been implemented, but the repository is not presently an out-of-the-box replication:

- No trained models or generated results are included.
- Training and analysis use hard-coded Linux paths under `/home/guetlid95/...`.
- The current training notebook saves algorithms as `Predictive` and `Contrastive`, while the analysis notebook expects directory names containing `Predictive_Prior` and `Forward_Forward`. Those paths will not match.
- The experiment-grid cell currently only prints conditions because its `run_experiment(...)` call is commented out.
- The active configuration trains only one `Video Learning` supervised model unless the notebook is manually changed.
- The requirements file looks like a complete Google Colab environment export, including Linux/CUDA-specific packages, rather than a minimal project dependency list. Installing it directly on Windows is unlikely to work.
- The analysis notebook uses `stats.wilcoxon` without importing `scipy.stats`.
- Ridge regression produces \(R^2\), but the notebook calculates intervals and tests using formulas intended for classification accuracy. That part should be corrected before relying on its statistical conclusions.
- There are no automated tests or command-line entry point.

So, the main research ideas and computational components are present, but some cleanup is needed to make the project reproducible on this machine and to ensure that all statistical analyses are methodologically consistent.