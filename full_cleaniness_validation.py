import math
import re
import statistics
import string
from collections import Counter

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


# ---------- Loader ----------

def load_pdfs(path: str = "."):
    """
    Load all PDFs recursively from the given path into LangChain Documents.
    """
    loader = DirectoryLoader(
        path=path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    print(f"[LOAD] Loaded {len(docs)} documents from {path}")
    return docs


# ---------- Metrics / helpers ----------

def shannon_entropy(text: str) -> float:
    """
    Estimate Shannon entropy of the given text.
    Lower values can indicate low-information / garbage (e.g., OCR noise).
    """
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    probs = [count / total for count in freq.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def is_non_ascii_heavy(text: str, threshold_ratio: float = 0.05) -> bool:
    """
    Returns True if more than threshold_ratio of characters are non-ASCII.
    """
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) > threshold_ratio


def punctuation_density(text: str) -> float:
    """
    Ratio of punctuation characters to total length.
    Very low density in long text can indicate bad OCR or broken text.
    """
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in string.punctuation)
    return punct / len(text)


def newline_count(text: str) -> int:
    return text.count("\n")


# ---------- Main validator ----------

def validate_docs(docs):
    """
    Run a series of heuristic checks on a list of LangChain Documents and
    print a human-readable cleanliness report.
    """

    print("\n========== CLEANLINESS VALIDATION ==========")

    if not docs:
        print("[WARN] No documents loaded; nothing to validate.")
        return

    # ---- Basic length stats ----
    lengths = [len(d.page_content or "") for d in docs]
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    median_len = statistics.median(lengths)

    print("\n[STATS] Chunk length (characters)")
    print(f"  Count : {len(docs)}")
    print(f"  Min   : {min_len}")
    print(f"  Max   : {max_len}")
    print(f"  Avg   : {avg_len:.1f}")
    print(f"  Median: {median_len:.1f}")

    # Flag extremes
    very_short = [i for i, L in enumerate(lengths) if L < 50]
    very_long = [i for i, L in enumerate(lengths) if L > 8000]

    if very_short:
        print(f"\n[WARN] {len(very_short)} chunks are very short (< 50 chars). "
              f"Example indices: {very_short[:10]}")
    if very_long:
        print(f"[WARN] {len(very_long)} chunks are very long (> 8000 chars). "
              f"Example indices: {very_long[:10]}")

    # ---- Duplicate content detection ----
    print("\n[DUPLICATES] Checking for repeated chunks (exact duplicates)...")
    content_counts = Counter(d.page_content for d in docs)
    repeated = [(txt, c) for txt, c in content_counts.items() if c > 3]

    if not repeated:
        print("  No highly repeated chunks found (good).")
    else:
        print(f"  Found {len(repeated)} chunk texts that repeat > 3 times.")
        print("  This may indicate headers/footers or boilerplate not removed.")
        # Show a few examples
        for i, (txt, c) in enumerate(sorted(repeated, key=lambda x: -x[1])[:5]):
            preview = txt.replace("\n", " ")[:120]
            print(f"    #{i+1} repeat_count={c} preview='{preview}...'")

    # ---- Entropy analysis ----
    print("\n[ENTROPY] Checking for low-entropy (possible OCR garbage) chunks...")
    entropies = [shannon_entropy(d.page_content or "") for d in docs]
    low_entropy_indices = [i for i, e in enumerate(entropies) if e > 0 and e < 2.5]

    if low_entropy_indices:
        print(f"  {len(low_entropy_indices)} chunks have low entropy (< 2.5). "
              f"Example indices: {low_entropy_indices[:10]}")
    else:
        print("  No obviously low-entropy chunks detected.")

    # ---- Weird character / OCR artifact checks ----
    print("\n[CHARS] Checking for weird / non-ASCII / OCR characters...")

    ligature_pattern = re.compile(r"[ﬁﬂ]")
    replacement_char_pattern = re.compile(r"[�]")
    hyphen_break_pattern = re.compile(r"-\n")

    ligature_hits = 0
    replacement_hits = 0
    hyphen_break_hits = 0
    non_ascii_heavy_hits = 0

    for d in docs:
        text = d.page_content or ""
        if ligature_pattern.search(text):
            ligature_hits += 1
        if replacement_char_pattern.search(text):
            replacement_hits += 1
        if hyphen_break_pattern.search(text):
            hyphen_break_hits += 1
        if is_non_ascii_heavy(text):
            non_ascii_heavy_hits += 1

    if ligature_hits:
        print(f"  [WARN] {ligature_hits} chunks contain ligature characters (ﬁ, ﬂ).")
    if replacement_hits:
        print(f"  [WARN] {replacement_hits} chunks contain the replacement char '�'.")
    if hyphen_break_hits:
        print(f"  [WARN] {hyphen_break_hits} chunks contain hyphen line-break patterns ('-\\n').")
    if non_ascii_heavy_hits:
        print(f"  [WARN] {non_ascii_heavy_hits} chunks have a high proportion of non-ASCII characters.")

    if not any([ligature_hits, replacement_hits, hyphen_break_hits, non_ascii_heavy_hits]):
        print("  No major weird-character issues detected.")

    # ---- Newline + punctuation diagnostics ----
    print("\n[STRUCTURE] Checking newline and punctuation patterns...")

    newline_counts = [newline_count(d.page_content or "") for d in docs]
    high_newline_indices = [i for i, n in enumerate(newline_counts) if n > 30]

    if high_newline_indices:
        print(f"  [WARN] {len(high_newline_indices)} chunks have many newlines (> 30). "
              f"Example indices: {high_newline_indices[:10]}")

    punct_densities = [punctuation_density(d.page_content or "") for d in docs]
    low_punct_indices = [
        i for i, (dens, L) in enumerate(zip(punct_densities, lengths))
        if L > 300 and dens < 0.001  # long text, almost no punctuation
    ]

    if low_punct_indices:
        print(f"  [WARN] {len(low_punct_indices)} chunks are long but have almost no punctuation. "
              f"May indicate poor extraction / OCR. Example indices: {low_punct_indices[:10]}")
    else:
        print("  Punctuation patterns look normal for most chunks.")

    # ---- Empty / near-empty chunks ----
    empty_indices = [i for i, L in enumerate(lengths) if L < 5]
    if empty_indices:
        print(f"\n[WARN] {len(empty_indices)} chunks are empty or near-empty (< 5 chars). "
              f"Example indices: {empty_indices[:10]}")

    print("\n========== VALIDATION COMPLETE ==========\n")
    print("Interpretation:")
    print("  • A few warnings are normal.")
    print("  • Many repeated chunks → consider removing headers/footers.")
    print("  • Low-entropy / low-punctuation / non-ASCII-heavy chunks → watch for OCR issues.")
    print("  • Very short/long chunks → consider re-splitting or cleaning upstream.")


# ---------- Entry point ----------

if __name__ == "__main__":
    docs = load_pdfs(path=".")
    validate_docs(docs)


print(docs[8].page_content)
print(docs[16].page_content)