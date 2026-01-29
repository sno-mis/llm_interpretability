#!/usr/bin/env python3
"""Fix script for notebook 04_mechanistic_interp.ipynb

Fixes:
1. HIGH: Negative name movers description - they boost the S (subject) token,
   working opposite to name movers, NOT performing S-inhibition.
2. HIGH: Ensure to_single_token calls for names use space prefix for GPT-2 tokenization.
3. MEDIUM: Mark redundant first forward pass in attribution patching as dead code.
4. MEDIUM: Add comment about all-position patching being a simplification.
"""

import json
import re
import sys

NOTEBOOK_PATH = "notebooks/04_mechanistic_interp.ipynb"

def load_notebook(path):
    with open(path, "r") as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

def get_source(cell):
    return "".join(cell["source"])

def set_source_from_string(cell, new_src):
    """Set cell source from a single string, splitting into lines for notebook format."""
    lines = new_src.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]]
    if lines:
        cell["source"].append(lines[-1])

def fix_negative_name_movers(nb):
    """Fix 1: Correct the description of negative name movers."""
    changes = 0
    for cell in nb["cells"]:
        src = get_source(cell)
        # Find the incorrect description
        old = "**Negative name movers**: Suppress the direct object (S-inhibition)"
        new = "**Negative name movers**: Boost the subject (S) token in the logits, acting opposite to name movers"
        if old in src:
            src = src.replace(old, new)
            set_source_from_string(cell, src)
            changes += 1
            print(f"  Fixed negative name movers description in cell {cell.get('id', '?')}")
    return changes

def fix_to_single_token_space_prefix(nb):
    """Fix 2: Ensure to_single_token calls for names have space prefix."""
    changes = 0
    name_pattern = re.compile(r'to_single_token\("([A-Z][a-z]+)"\)')
    for cell in nb["cells"]:
        src = get_source(cell)
        matches = list(name_pattern.finditer(src))
        if matches:
            for m in reversed(matches):
                name = m.group(1)
                old_call = m.group(0)
                new_call = f'to_single_token(" {name}")'
                if old_call != new_call:
                    src = src[:m.start()] + new_call + src[m.end():]
                    changes += 1
                    print(f"  Fixed to_single_token(\"{name}\") -> to_single_token(\" {name}\") in cell {cell.get('id', '?')}")
            if changes:
                set_source_from_string(cell, src)
    if changes == 0:
        print("  No to_single_token calls missing space prefix (already correct)")
    return changes

def fix_dead_code_attribution(nb):
    """Fix 3: Mark or remove redundant first forward pass in attribution patching."""
    changes = 0
    for cell in nb["cells"]:
        src = get_source(cell)
        # Look for the redundant forward pass before the hooks-based pass
        old_block = (
            "# Forward pass with gradient tracking\n"
            "corrupted_logits_grad = model(corrupted_tokens)\n"
            "logit_diff = corrupted_logits_grad[0, -1, mary_token] - corrupted_logits_grad[0, -1, alice_token]"
        )
        if old_block in src:
            new_block = (
                "# NOTE: The forward pass below is redundant (dead code) -- the run_with_hooks call\n"
                "# further down repeats this computation with gradient hooks attached.\n"
                "# corrupted_logits_grad = model(corrupted_tokens)\n"
                "# logit_diff = corrupted_logits_grad[0, -1, mary_token] - corrupted_logits_grad[0, -1, alice_token]"
            )
            src = src.replace(old_block, new_block)
            set_source_from_string(cell, src)
            changes += 1
            print(f"  Commented out redundant forward pass in cell {cell.get('id', '?')}")
    return changes

def fix_all_position_patching_comment(nb):
    """Fix 4: Add comment about all-position patching being a simplification."""
    changes = 0
    for cell in nb["cells"]:
        src = get_source(cell)
        # Target the Section 3 head patching cell (a2b3c4d5) which patches all positions
        target_line = "activation[0, :, h, :] = clean_cache[f\"blocks.{l}.attn.hook_result\"][0, :, h, :]"
        if target_line in src and cell.get("id") == "a2b3c4d5":
            old = (
                "        def patch_head_output(activation, hook, l=layer, h=head):\n"
                "            activation[0, :, h, :] = clean_cache[f\"blocks.{l}.attn.hook_result\"][0, :, h, :]\n"
                "            return activation"
            )
            new = (
                "        def patch_head_output(activation, hook, l=layer, h=head):\n"
                "            # NOTE: This patches all sequence positions at once, which is a simplification.\n"
                "            # Position-specific patching (e.g., only at the last token or IO position)\n"
                "            # would give more precise results about where each head matters.\n"
                "            activation[0, :, h, :] = clean_cache[f\"blocks.{l}.attn.hook_result\"][0, :, h, :]\n"
                "            return activation"
            )
            if old in src:
                src = src.replace(old, new)
                set_source_from_string(cell, src)
                changes += 1
                print(f"  Added position-specific patching comment in cell {cell.get('id', '?')}")
    return changes

def main():
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    nb = load_notebook(NOTEBOOK_PATH)
    total_changes = 0

    print("\nFix 1: Correct negative name movers description")
    total_changes += fix_negative_name_movers(nb)

    print("\nFix 2: Ensure to_single_token space prefix for names")
    total_changes += fix_to_single_token_space_prefix(nb)

    print("\nFix 3: Comment out redundant forward pass (dead code)")
    total_changes += fix_dead_code_attribution(nb)

    print("\nFix 4: Add comment about all-position patching simplification")
    total_changes += fix_all_position_patching_comment(nb)

    if total_changes > 0:
        save_notebook(nb, NOTEBOOK_PATH)
        print(f"\nSaved {total_changes} fix(es) to {NOTEBOOK_PATH}")
    else:
        print("\nNo changes needed.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
