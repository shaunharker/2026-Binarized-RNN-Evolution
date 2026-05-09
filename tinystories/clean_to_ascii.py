import sys
import argparse
import os

def clean_to_strict_ascii(input_path, output_path):
    """
    Reads a text file, translates smart punctuation to ASCII, 
    and removes all remaining non-ASCII (byte > 127) characters.
    """
    # Common Unicode punctuation we want to save by converting to ASCII equivalents
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote / apostrophe
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2013': "-",   # En dash
        '\u2014': "--",  # Em dash
        '\u2026': "...", # Ellipsis
        '\u00a0': " ",   # Non-breaking space
        '\u00e9': "e",   # é (common in words like café)
    }

    try:
        file_size = os.path.getsize(input_path)
        processed_bytes = 0
        
        print(f"Reading: {input_path}")
        print(f"Writing strictly ASCII text to: {output_path}...")
        
        # Open input allowing utf-8 but ignoring catastrophic encoding errors
        # Open output forcing strict ascii
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_path, 'w', encoding='ascii') as outfile:
             
             for line in infile:
                 # 1. Translate the common smart punctuation
                 for search_char, replace_char in replacements.items():
                     line = line.replace(search_char, replace_char)
                 
                 # 2. Force string into strict ASCII (bytes 0-127)
                 # errors='ignore' ruthlessly deletes anything > 127
                 clean_bytes = line.encode('ascii', errors='ignore')
                 
                 # 3. Decode back to a string and write to the new file
                 clean_line = clean_bytes.decode('ascii')
                 outfile.write(clean_line)

                 # Simple progress tracker (prints every ~10MB)
                 processed_bytes += len(line)
                 if processed_bytes % 10_000_000 < len(line):
                     percent = (processed_bytes / file_size) * 100
                     print(f"Progress: {percent:.1f}%", end='\r')

        print("\n\nDone! File successfully converted to Strict ASCII.")
        print("Maximum possible unique bytes in the new file: 128.")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert a dataset strictly to ASCII, dropping all weird bytes.")
    parser.add_argument("input", help="Path to the original dirty text file")
    parser.add_argument("output", help="Path to save the cleaned text file")
    
    args = parser.parse_args()
    clean_to_strict_ascii(args.input, args.output)

if __name__ == "__main__":
    main()
