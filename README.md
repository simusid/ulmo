# ulmo
Large acoustic models

# Things to investigate
- alternatives to K-means and better tokenization
  - Self supervised pretext tasks such as next frame prediction, infill, ABX (STFT, Mel Spec, CQT, CWT, DWT, WPT choice tasks)
- long context to discover repeated impulses
- adding an \<UNK\> token 
- Try a denoising autoencoder in place of kmeans to create labels.
- Data augmentation by using sets of spectrogram parameters (vary SR +- 10%)
- methods of self supervision such as SimCLR, wav2vec2,and BYOL

# detailed OpenAI prompt
Develop a Vector Quantized Variational Autoencoder (VQ-VAE) using Keras to tokenize acoustic spectrograms, following these detailed requirements and specifications:
1. Input Specifications

    Accept model inputs as batches of log-normalized numpy arrays with dtype float32.
    Each input tensor should have shape (B, M, M, 1):
        B: batch size (arbitrary, to be set during training/evaluation)
        M: spectrogram width/height, default to 224; must be configurable.

2. Encoder Architecture

    Construct the encoder as a stack of E convolutional layers (default E=4).
    Use best practices for CNN design (appropriate kernel sizes, strides, activations, and normalization typical for spectrogram-like image data).
    The encoder should reduce dimensionality as appropriate for subsequent quantization.

3. Embedding & Code Book Layer

    Add an embedding (vector quantization) layer after the encoder:
        Embedding dimension per code: EMB (default 32).
        Code book size: CB (default 512).
        The code book must be shared globally (across all inputs and batches), so the encoder's outputs can be tokenized identically at inference.
    Implement code book updating with an exponential moving average (EMA) mechanism; make EMA coefficient configurable.
    Ensure quantization and code book lookup mechanisms are compatible with Keras.

4. Decoder Architecture

    Implement the decoder as a stack of D convolutional layers (default D=4).
    Design the decoder so that its output reconstructs inputs exactly to the shape (B, M, M, 1).

5. Output and Shape Validation

    Validate that the model's final output for each batch matches the input shape (B, M, M, 1).
    Include explicit assertion tests for output shape after reconstruction.

6. Loss Functions

    Compute and record the following losses for each batch:
        Reconstruction Loss (e.g. mean squared error) between input and output.
        Code Book Commitment Loss to enforce effective use of the code book.
        Any other standard losses necessary for VQ-VAE stability and performance.
    Loss values must be captured and made accessible per batch/epoch.
    Integrating TensorBoard logging for loss curves is encouraged but optional.

7. Model Configuration & Parameterization

    All core parameters (M, E, EMB, CB, D, EMA coefficient) should be configurable with clearly established default values, e.g., via function arguments or constructor parameters.

8. Project Structure and Deliverables

    Organize code into clear modules or classes for encoder, vector quantization layer, decoder, and training/evaluation routines.
    Supply scripts or functions for:
        Model training (including configuration of parameters and input data).
        Evaluation on held-out or test-data batches.
        Asserting output shape and code book updating.
    Documentation is not required initially, but include inline code comments for clarity where appropriate.
    Provide a short usage example (e.g., in a Python docstring or README stub) showing instantiation, input preprocessing, and output shape checking.

Example of Improved Instruction:

    “Configure the EMA code book updating process with a float parameter ema_coeff (default: 0.99). Ensure this is accessible for adjustment when instantiating the model or embedding layer.”

Summary Table of Required Default Parameters:
Parameter	Description	Default Value
M	Input height/width	224
E	Encoder conv layers	4
EMB	Embedding dimension	32
CB	Code book size	512
D	Decoder conv layers	4
EMA coefficient	Code book updating rate	0.99

(Adjust table as needed.)

Notes:

    Input data must be provided as batched, log-normalized, float32 numpy arrays.
    Code book is shared for all batches and used for both training and inference as a quantization vocab.
    Focus on robust and clear model parameterization; documentation may be developed after initial project success.
    Provide batch-wise loss capture; TensorBoard integration is optional, not required.

If any new ambiguities arise, seek further clarification before starting implementation.
