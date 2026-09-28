"""Synonym Graph Filter Example — token expansion with synonyms.

Mirrors the `synonym_graph_filter.rs` Rust example.

Demonstrates:
- Creating a synonym dictionary with bidirectional synonym groups
- Tokenizing text with WhitespaceTokenizer
- Applying SynonymGraphFilter without boost
- Applying SynonymGraphFilter with boost < 1.0 (synonyms weighted down)

Run with:
    maturin develop
    python examples/synonym_graph_filter.py
"""

import laurus


def main() -> None:
    print("=== SynonymGraphFilter Usage Example ===\n")

    # ── Step 1: Create synonym dictionary ─────────────────────────────────
    print("Step 1: Creating synonym dictionary")
    syn_dict = laurus.SynonymDictionary()
    syn_dict.add_synonym_group(["venv", "virtual environment"])
    syn_dict.add_synonym_group(["pip", "package installer"])

    print("  Added synonyms:")
    print("    - 'venv' <-> 'virtual environment'")
    print("    - 'pip' <-> 'package installer'\n")

    tokenizer = laurus.WhitespaceTokenizer()

    # ── Step 2: Apply filter WITHOUT boost ────────────────────────────────
    print("Step 2: Applying filter WITHOUT boost")
    filt = laurus.SynonymGraphFilter(syn_dict, keep_original=True, boost=1.0)

    input_text = "venv tutorial"
    print(f'  Input: "{input_text}"\n')

    tokens = tokenizer.tokenize(input_text)
    result_tokens = filt.apply(tokens)

    print("  Output tokens:")
    for i, tok in enumerate(result_tokens):
        print(
            f"    [{i}] {tok.text!r:20s}  "
            f"pos={tok.position}  "
            f"pos_inc={tok.position_increment}  "
            f"pos_len={tok.position_length}  "
            f"boost={tok.boost:.2f}"
        )

    print()
    print("  Explanation:")
    print("    - All tokens have boost=1.0 (default)")
    print("    - Synonyms have equal weight to original tokens\n")


    # ── Step 3: Apply filter WITH boost=0.8 ───────────────────────────────
    print("Step 3: Applying filter WITH boost=0.8")
    filt_boosted = laurus.SynonymGraphFilter(syn_dict, keep_original=True, boost=0.8)

    print(f'  Input: "{input_text}"\n')

    tokens = tokenizer.tokenize(input_text)
    result_tokens = filt_boosted.apply(tokens)

    print("  Output tokens:")
    for i, tok in enumerate(result_tokens):
        print(
            f"    [{i}] {tok.text!r:20s}  "
            f"pos={tok.position}  "
            f"pos_inc={tok.position_increment}  "
            f"pos_len={tok.position_length}  "
            f"boost={tok.boost:.2f}"
        )

    print()
    print("  Explanation:")
    print("    - Original token 'venv' has boost=1.0")
    print("    - Synonym tokens have boost < 1.0")
    print("    - Lower boost means synonyms contribute less to the final score")
    print("    - This helps prioritize exact matches over synonym matches\n")

    # ── Step 4: Multi-word synonym expansion ──────────────────────────────
    print("Step 4: Multi-word synonym expansion (pip -> package installer)")
    filt2 = laurus.SynonymGraphFilter(syn_dict, keep_original=True, boost=0.9)

    input_text2 = "pip install"
    print(f'  Input: "{input_text2}"\n')

    tokens2 = tokenizer.tokenize(input_text2)
    result_tokens2 = filt2.apply(tokens2)

    print("  Output tokens (position graph):")
    for i, tok in enumerate(result_tokens2):
        print(
            f"    [{i}] {tok.text!r:25s}  "
            f"pos={tok.position}  "
            f"pos_inc={tok.position_increment}  "
            f"pos_len={tok.position_length}  "
            f"boost={tok.boost:.2f}"
        )

    print()
    print("  Explanation:")
    print("    - 'package installer' spans positions 0 and 1")
    print("    - position_length 2 on 'pip' makes it span both words of the synonym")
    print("    - Downstream phrase indexing uses this graph to support phrase queries\n")

    print("Use cases for boost:")
    print("  - boost=0.8: Synonyms have 80% weight (common default)")
    print("  - boost=0.5: Synonyms have 50% weight (conservative)")
    print("  - boost=1.0: Synonyms equal to originals (no adjustment)\n")

    print("SynonymGraphFilter example completed!")


if __name__ == "__main__":
    main()
