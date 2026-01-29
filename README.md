# LLM Interpretability: From Foundations to Frontier

A self-updating study guide for understanding state-of-the-art LLM interpretability — in theory and practice.

## Structure

| # | Notebook | Topic |
|---|----------|-------|
| 00 | Guide Overview | Field map, reading order, key papers |
| 01 | Transformer Circuits | QK/OV decomposition, residual stream, composition |
| 02 | Superposition | Polysemanticity, toy models, feature geometry |
| 03 | Sparse Autoencoders | Dictionary learning, SAE training, evaluation |
| 04 | Activation Patching | Causal tracing, circuit discovery, IOI case study |
| 05 | Logit Lens | Intermediate predictions, tuned lens |
| 06 | Probing | Linear probes, geometry of truth, CCS |
| 07 | Representation Engineering | Steering vectors, activation addition |
| 08 | Circuit Tracing | Attribution graphs, Anthropic 2025 methods |
| 09 | Tools & Practice | TransformerLens, SAELens, Neuronpedia, pyvene |
| 00b | From Scratch | Raw PyTorch interpretability on a tiny transformer |
| 10 | Open Problems | Frontier questions, research directions |
| 11 | Automated Interpretability | LLM-based feature labeling, scoring protocols, auto-interp pipelines |
| 12 | Developmental Interpretability | Watch circuits emerge during training |
| 13 | Capstone Project | Find a circuit end-to-end |

## Features

- **Running example**: An IOI (Indirect Object Identification) investigation is threaded through every notebook, so you see the same phenomenon analyzed from each technique's perspective.
- **Exercises**: Each notebook includes hands-on exercises with hints, so you can test your understanding.
- **Deep math**: Notebooks 01–03 include rigorous mathematical derivations (information-theoretic bounds, polytope geometry, loss landscape analysis).
- **Build it yourself first**: Notebook 00b implements interpretability from raw PyTorch — no libraries, just matrix math.
- **Interactive visualizations**: Plotly-powered heatmaps and charts with hover data throughout notebooks 01, 04, 05.
- **Hacker tone**: Direct, Karpathy-style prose — no passive voice, no filler.
- **Self-updating tracker**: Automated paper fetching with TF-IDF relevance scoring and guide update suggestions.

## Self-Updating Research Tracker

The `research_tracker/` directory contains scripts that automatically fetch new interpretability papers from arXiv and Semantic Scholar, score them for relevance, and generate digest notebooks in `digests/`.

```bash
# Fetch new papers
python research_tracker/tracker.py

# Generate a digest notebook from recent papers
python research_tracker/digest.py
```

## Setup

```bash
pip install -r requirements.txt

# Optional: install from source for latest versions
pip install git+https://github.com/stanfordnlp/pyvene.git
pip install git+https://github.com/ndif-team/nnsight.git
```

## Prerequisites

- Strong ML foundations (transformers, PyTorch, backprop)
- Familiarity with key interpretability concepts (attention, features, superposition)
- GPU recommended for notebooks 03, 04, 07, 09
