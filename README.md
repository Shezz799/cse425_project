# Music Generation Project (Tasks 1-3)

This repo implements Tasks 1-3 for the Unsupervised Multi-Genre Music Generation project using MAESTRO MIDI data.

## Notes
- The project requirements call for multi-genre data in Task 2. This implementation uses MAESTRO only (classical piano) and explicitly documents this limitation in evaluation.
- The code is designed to run locally or in Kaggle notebooks. In Kaggle, use `/kaggle/input` for data and `/kaggle/working` for outputs.
- MIDI files link : https://drive.google.com/drive/folders/1vk2NSHYfiQ9ejYwcrjnle9R9H-smgWpx?usp=sharing


## Quickstart (Local)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Preprocess MAESTRO piano-roll windows and tokens:
   ```bash
   python -m src.preprocessing.pianoroll
   python -m src.preprocessing.tokens
   ```
3. Train models:
   ```bash
   python -m src.training.train_ae
   python -m src.training.train_vae
   python -m src.training.train_transformer
   ```
4. Generate samples and metrics:
   ```bash
   python -m src.generation.generate_music
   python -m src.evaluation.metrics
   ```

5. VAE latent interpolation (Task 2 deliverable):
   ```bash
   python -m src.generation.latent_interpolation --checkpoint outputs/checkpoints/lstm_vae.pt --num-samples 8
   ```

## Kaggle Notes
- Add MAESTRO dataset to the notebook and point `DATA_ROOT` to the dataset folder if auto-detection fails.
- Outputs are written under `/kaggle/working/outputs` by default.

## Project Layout
- `src/preprocessing`: MIDI loading, piano-roll creation, and tokenization
- `src/models`: LSTM autoencoder, VAE, and Transformer
- `src/training`: training loops and loss plotting
- `src/generation`: MIDI export and sample generation
- `src/evaluation`: metrics and baseline generators
- `notebooks`: Kaggle-ready notebooks for preprocessing and training
-
