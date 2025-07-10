from nltk.tokenize import RegexpTokenizer

def compute_avg_word_count(df, content_column="document", print_stats=True):
    # Create a German-friendly tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    def count_words_safe(doc):
        """
        Tokenizes text and counts words.
        """
        if isinstance(doc, dict):
            text = doc.get("content", "")
        else:
            text = str(doc) if doc is not None else ""
            
        tokens = tokenizer.tokenize(text) if text else []
        return len(tokens)
    
    # Calculate word counts
    df["word_count"] = df[content_column].apply(count_words_safe)
    
    # Compute average
    avg_words = df["word_count"].mean()
    
    if print_stats:
        print(f"Average words per document: {avg_words:.2f}")
        print(f"Min words: {df['word_count'].min()}")
        print(f"Max words: {df['word_count'].max()}")
    
    return avg_words