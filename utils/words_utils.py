from nltk.tokenize import RegexpTokenizer

def compute_avg_word_count(
    df,
    content_column: str = "document",
    print_stats: bool = True,
    in_place: bool = True,
):
    """
    Compute the average number of words per row for a column that may contain
    Haystack Documents, dicts with a "content" key, or raw strings.

    - Prefers `obj.content` when available (Haystack Document) to avoid
      accidentally counting stringified metadata/IDs.
    - Falls back to dict["content"], else string-cast safely.
    - Uses a Unicode-aware word tokenizer suitable for German text (\w+).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    content_column : str
        Column name containing the content objects (default: "document").
    print_stats : bool
        Whether to print average/min/max statistics.
    in_place : bool
        If True, store the computed counts in df["word_count"].

    Returns
    -------
    float
        The mean word count across rows.
    """

    tokenizer = RegexpTokenizer(r"\w+")

    def _extract_text(val) -> str:
        # Haystack Document objects typically expose .content
        if hasattr(val, "content") and isinstance(getattr(val, "content"), str):
            return val.content
        # Dict payload with content
        if isinstance(val, dict):
            return str(val.get("content", ""))
        # Fallback: best-effort string
        return str(val) if val is not None else ""

    def _count_words(val) -> int:
        text = _extract_text(val)
        if not text:
            return 0
        # Tokenize words (Unicode letters/digits/underscore)
        return len(tokenizer.tokenize(text))

    counts = df[content_column].apply(_count_words)
    if in_place:
        df["word_count"] = counts

    avg_words = float(counts.mean()) if len(counts) else 0.0

    if print_stats:
        print(f"Average words per document: {avg_words:.2f}")
        print(f"Min words: {int(counts.min()) if len(counts) else 0}")
        print(f"Max words: {int(counts.max()) if len(counts) else 0}")

    return avg_words