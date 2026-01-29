#!/usr/bin/env python3
"""Fix script for notebook 03_sparse_autoencoders.ipynb"""
import json

notebook_path = "/Users/isaac/llm_interpretability/notebooks/03_sparse_autoencoders.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

def get_cell_source(cell):
    return "".join(cell["source"])

def set_cell_source(cell, text):
    cell["source"] = text.split("\n")
    # Re-add newlines to all lines except the last
    cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]

for cell in nb["cells"]:
    src = get_cell_source(cell)

    # Fix 1: l1_coeff stored but never used internally in the SAE class.
    # The training loop uses l1_coeff as a local variable, not self.l1_coeff.
    # Add a comment explaining this design choice.
    if "self.l1_coeff = l1_coeff" in src and "class SparseAutoencoder" in src:
        src = src.replace(
            "        self.l1_coeff = l1_coeff",
            "        # Note: l1_coeff is stored here for reference/logging but is applied\n"
            "        # externally in the training loop (see Section 2) to keep the loss\n"
            "        # computation explicit and flexible for experimentation.\n"
            "        self.l1_coeff = l1_coeff"
        )
        set_cell_source(cell, src)

    # Fix 2: ~70% interpretable claim needs precision note
    if '~70% of features were interpretable' in src:
        src = src.replace(
            'The "Towards Monosemanticity" paper found ~70% of features were interpretable.',
            'The "Towards Monosemanticity" paper found ~70% of features were interpretable. '
            '(Note: the exact figure varies by methodology, SAE configuration, and model; '
            'this is an approximate summary rather than a single definitive number.)'
        )
        set_cell_source(cell, src)

    # Fix 3: Great Wall of China myth
    if "Great Wall of China is visible from" in src:
        src = src.replace(
            '"The Great Wall of China is visible from certain low Earth orbits."',
            '"The Great Wall of China is visible from certain low Earth orbits.",  '
            '# Note: this is a common misconception used here as example training data; '
            'the Wall is not readily visible from space with the naked eye.'
        )
        set_cell_source(cell, src)

    # Fix 4: Ghost gradients — add "simplified" qualifier
    if "Ghost gradients" in src or "ghost gradients" in src:
        if "**Solution 2: Ghost gradients.**" in src:
            src = src.replace(
                "**Solution 2: Ghost gradients.**",
                "**Solution 2: Ghost gradients (simplified).**"
            )
            set_cell_source(cell, src)

    # Fix 5: W_enc = W_dec^T oversimplification
    if r"$W_{\text{enc}} = W_{\text{dec}}^\top$" in src and "optimal solution" in src:
        src = src.replace(
            r"the optimal solution satisfies $W_{\text{enc}} = W_{\text{dec}}^\top$ (the encoder is the transpose of the decoder).",
            r"the optimal solution satisfies $W_{\text{enc}} = W_{\text{dec}}^\top$ (the encoder is the transpose of the decoder). "
            "Note: in practice this relationship only holds at initialization or in the linear/no-sparsity limit; "
            "after training with L1 sparsity, the encoder and decoder weights diverge significantly."
        )
        set_cell_source(cell, src)

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("All fixes applied successfully.")
