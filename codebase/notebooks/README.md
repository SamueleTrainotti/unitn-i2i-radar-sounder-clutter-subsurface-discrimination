# Notebooks

This directory contains Jupyter notebooks used for data exploration, visualization, and thesis figure generation. They are designed to be run interactively to analyze data and check intermediate outcomes without needing to execute full scripts.

## Contents

### `generate_thesis_plots.ipynb`

This is the primary notebook used to generate static statistical and structural plots for the thesis document. It heavily leverages the core logic inside `src/dataset/` to load and parse `.img` and `.xml` radar files to ensure maximum parity with the actual training loops.

**Key Visualizations:**
1.  **Raw Radargram Comparison:** Visualizes the full, unprocessed Real and Simulated radargrams side-by-side using a standard colormap.
2.  **Linear vs. dB Scale:** Compares radargrams parsed in raw linear intensity versus Decibel (dB) scale to show how we compress the dynamic range of radar signals for neural network ingestion.
3.  **A-Scan (Trace) Extraction:** Extracts and plots individual A-Scans (vertical traces representing signal intensity across depth) to inspect specific details in the waveform.
4.  **Signal Attenuation vs Depth:** Analyzes the average intensity of the signal (in dB) across all A-scans in the radargram, plotting depth against average intensity. Shows the physical decay of the radar signal through the ice, crucial for both real and simulated data comparison.
5.  **Patch Extraction Schematic:** A detailed visual explanation of how the sliding-window patching algorithm works. It plots a zoomed radargram section and highlights the Bounding Box Extraction logic (Width, Height, Overlap, and Surface/Center anchoring) with custom annotations.
6.  **Patch Size Field of View:** Provides a side-by-side illustration of extracting patches of varying sizes (e.g., 64x64, 128x128, 256x256, 512x512) centered around the same anchor point on the radargram, demonstrating the context window passed to the network.

### `read_datagram.ipynb`
A utility notebook for inspecting the raw binary format of the `.img` files along with their XML metadata.

### `example.ipynb`
General playground for testing imports and PyTorch operations on the dataset.

## Usage

You can run these notebooks locally using Jupyter Lab or Jupyter Notebook.
Make sure your Python environment has access to the `src` folder (this is handled automatically if you run `pip install -e .` on the main project root or add `src` to your `PYTHONPATH`).
