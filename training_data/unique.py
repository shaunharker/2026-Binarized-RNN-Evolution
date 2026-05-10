import sys
import argparse
from collections import Counter

def get_byte_counts(filepath):
    """Reads a file in chunks and returns a Counter dictionary of byte frequencies."""
    byte_counts = Counter()
    
    try:
        # Open in binary read mode ('rb')
        with open(filepath, 'rb') as f:
            while True:
                # Read in 64KB chunks for efficient disk I/O on large files
                chunk = f.read(65536)
                if not chunk:
                    break
                
                # Counter.update() automatically tallies occurrences of each byte in the chunk
                byte_counts.update(chunk)
                
        return byte_counts
        
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when trying to read '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

def display_byte_counts(byte_counts):
    """Prints the bytes and their counts in a formatted table."""
    # Get a sorted list of the unique bytes (0-255)
    unique_bytes = sorted(byte_counts.keys())
    
    print(f"Found {len(unique_bytes)} unique bytes (out of 256 possible).\n")
    
    # Table Header
    print(f"{'Dec':<5} | {'Hex':<6} | {'Char':<6} | {'Count':<12} | {'Description'}")
    print("-" * 58)
    
    for b in unique_bytes:
        # Decimal, Hexadecimal, and Count representation
        dec_str = str(b)
        hex_str = f"0x{b:02X}"
        count_str = f"{byte_counts[b]:,}" # Adds commas to large numbers
        
        # Determine the ASCII character representation
        if 32 <= b <= 126:
            char_str = chr(b)
            desc = "Printable ASCII"
        elif b == 0:
            char_str = "NUL"
            desc = "Null Byte"
        elif b == 9:
            char_str = "\\t"
            desc = "Horizontal Tab"
        elif b == 10:
            char_str = "\\n"
            desc = "Line Feed (LF)"
        elif b == 13:
            char_str = "\\r"
            desc = "Carriage Return (CR)"
        elif b == 255:
            char_str = "."
            desc = "Non-breaking space / End"
        else:
            char_str = "."
            desc = "Non-printable"
            
        print(f"{dec_str:<5} | {hex_str:<6} | {char_str:<6} | {count_str:<12} | {desc}")

def main():
    parser = argparse.ArgumentParser(description="Count and display all unique bytes in a file.")
    parser.add_argument("filepath", help="Path to the file you want to analyze")
    args = parser.parse_args()

    # Get the counts
    counts = get_byte_counts(args.filepath)
    
    # Display them
    display_byte_counts(counts)

if __name__ == "__main__":
    main()
