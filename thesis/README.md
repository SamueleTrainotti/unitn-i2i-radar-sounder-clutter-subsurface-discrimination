# Master's Thesis: Image-to-Image GAN Translation for Radar Sounding

This directory contains the complete LaTeX source code for the Master's Thesis, which explores conditional Generative Adversarial Networks (Pix2Pix and CycleGAN) for unsupervised anomaly detection in planetary subsurface radar data (SHARAD).

## Directory Structure

The thesis adopts a highly modular structure to maintain clean separation of concerns and simplify version control:

- `samuele_trainotti_IE_2324.tex`: The main entry point file. It handles the `\documentclass`, imports packages, configures the frontend, and stitches together the chapters using `\subfile{}`.
- `preamble.tex`: Centralized definition of all packages, layout margins, custom commands, and bibliography configuration.
- `front.tex`: The title page configuration.
- `biblio.bib`: The BibTeX database containing all citations categorized by topic.
- `01_introduction.tex` through `08_appendices/`: Individual folders or files containing the chapter text. Note that several complex chapters (like Methodology and Results) are subdivided into dedicated standalone `.tex` files per section for easier editing and minimal merge conflicts.
- `images/`: High-resolution figures, plots, and architectural diagrams (SVG/PDF/PNG) exported by the main project.
- `samuele_trainotti_IE_2324.pdf`: The final compiled output manuscript.

## Compilation Instructions

To compile the thesis locally, it is heavily recommended to use `pdflatex` coupled with `bibtex` for the citations. Ensure you compile everything from the root `thesis/` folder.

A standard full build sequence is:

```bash
pdflatex -interaction=nonstopmode samuele_trainotti_IE_2324.tex
bibtex samuele_trainotti_IE_2324
pdflatex -interaction=nonstopmode samuele_trainotti_IE_2324.tex
pdflatex -interaction=nonstopmode samuele_trainotti_IE_2324.tex
```

> **Note:** The thesis is structured such that every sentence begins on a new line. This prevents massive git diffs and merge conflicts when collaborating.

## Utility Scripts for Maintainers

During the writing of the thesis, specialized Python scripts were developed to automate complex LaTeX refactoring tasks. These scripts are included in this directory to assist authors or maintainers working with similar large documents.

### 1. `refactor_sentences.py`
LaTeX version control is drastically improved when each sentence sits on its own line. However, manually dividing paragraphs or naively splitting on periods (`.`) corrupts the document layout by breaking `\caption`s, equations, or common abbreviations (e.g., "e.g.", "Fig.").

This script uses recursive brace-counting and advanced regular expressions to safely refactor standard paragraphs into a "one sentence per line" structure.
- **Protection Logic**: It safely parses and ignores nested environments (`\caption{...}`, `\footnote{...}`, `\textbf{...}`), all inline and display math contexts (`$..$`, `$$..$$`, `\[..\]`), and comment blocks.
- **Abbreviation Awareness**: Prevents breaking strings like `et al.`, `i.e.`, `Eq.`, and `Fig.`.
- **Usage**:
  ```bash
  python refactor_sentences.py
  ```
  *The script will recursively scan all active `.tex` files in the directory and perform the splits in-place.*

### 2. `finalize_changes.py`
When heavily utilizing the `changes` LaTeX package (e.g., using `\added{...}`, `\deleted{...}`, `\replaced{new}{old}`), the source code can quickly become a dense, unreadable thicket of markup.

When the review phase is completed, `finalize_changes.py` resolves the document state recursively across all chapter subfiles.
- **Operation**: It unconditionally accepts all `\added` text (removing the tags), fully removes `\deleted` blocks and their content, and accepts the `new` portion of `\replaced{new}{old}` while dropping the `old` portion.
- **Protection**: Includes basic recursion to handle overlapping or multi-line usages while skipping backup and `.git` folders.
- **Usage**:
  ```bash
  python finalize_changes.py
  ```
