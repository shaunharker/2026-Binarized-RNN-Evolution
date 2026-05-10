import math
import sys
import argparse
from collections import Counter

def calculate_ngram_entropy(data: bytes, n: int) -> float:
    """
    Calculates the Shannon entropy of n-grams in a sequence of bytes.
    """
    if len(data) < n:
        return 0.0
        
    total_ngrams = len(data) - n + 1
    
    # We use a generator expression to extract n-grams to save memory
    ngram_counts = Counter(data[i:i+n] for i in range(total_ngrams))
    
    entropy = 0.0
    for count in ngram_counts.values():
        probability = count / total_ngrams
        entropy -= probability * math.log2(probability)
        
    return entropy

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Shannon entropy and first differences of n-grams (1-5) for a file."
    )
    # Default to training.txt if no file is provided
    parser.add_argument(
        "filename", 
        nargs="?", 
        default="training.txt", 
        help="Path to the text file (default: training.txt)"
    )
    
    args = parser.parse_args()
    
    try:
        # Read the file as a sequence of bytes
        with open(args.filename, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.filename}' was not found.")
        sys.exit(1)
        
    if not data:
        print("The file is empty.")
        sys.exit(0)
        
    print(f"Loaded '{args.filename}' ({len(data):,} bytes).")
    print("-" * 60)
    
    entropies =[]
    max_n = min(5, len(data)) # Ensures we don't look for 5-grams in a 3-byte file
    
    print("Shannon Entropies:")
    for n in range(1, max_n + 1):
        entropy = calculate_ngram_entropy(data, n)
        entropies.append(entropy)
        print(f"{n}-gram (H{n}): {entropy:8.4f} bits")
        
    print("-" * 60)
    print("First Differences (Average information in the n-th byte):")
    
    if entropies:
        # The information of the 1st byte is simply the 1-gram entropy
        print(f"1st byte (H1)      : {entropies[0]:8.4f} bits")
        
    # Calculate H_{n+1} - H_n
    for n in range(1, len(entropies)):
        diff = entropies[n] - entropies[n-1]
        print(f"{n+1}th byte (H{n+1} - H{n}): {diff:8.4f} bits")

if __name__ == "__main__":
    main()
