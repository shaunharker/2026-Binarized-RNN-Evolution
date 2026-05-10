import sys
import argparse

def display_context(window, target_pos_in_window, absolute_pos, start_pos):
    """Safely prints the surrounding bytes in both text and hex formats."""
    target_byte = window[target_pos_in_window]
    
    print("=" * 60)
    print(f" TARGET FOUND: {hex(target_byte)} (Dec: {target_byte})")
    print(f" Absolute File Offset: {absolute_pos} (0x{absolute_pos:X})")
    print("=" * 60)
    
    # --- METHOD 1: SAFE TEXT VIEW ---
    print("\n--- Surrounding Text (Safe View) ---")
    safe_text = ""
    for i, b in enumerate(window):
        if i == target_pos_in_window:
            safe_text += f"\n\n >>>[ TARGET BYTE: {hex(b)} ] <<< \n\n"
        else:
            # Printable ASCII or standard whitespace (Tab, LF, CR)
            if 32 <= b <= 126 or b in (9, 10, 13):
                safe_text += chr(b)
            else:
                # Escape other weird bytes safely
                safe_text += f"\\x{b:02x}"
                
    print(safe_text)
    print("\n" + "-" * 60 + "\n")
    
    # --- METHOD 2: HEX DUMP VIEW ---
    print("--- Surrounding Hex Dump ---")
    for i in range(0, len(window), 16):
        chunk = window[i:i+16]
        
        # Format Hex part: standard bytes get spaces " XX ", target gets brackets "[XX]"
        hex_parts =[]
        for j, b in enumerate(chunk):
            if i + j == target_pos_in_window:
                hex_parts.append(f"[{b:02X}]")
            else:
                hex_parts.append(f" {b:02X} ")
        hex_str = "".join(hex_parts)
        
        # Format ASCII part: standard chars are normal, target gets brackets
        ascii_parts =[]
        for j, b in enumerate(chunk):
            char = chr(b) if 32 <= b <= 126 else "."
            if i + j == target_pos_in_window:
                ascii_parts.append(f"[{char}]")
            else:
                ascii_parts.append(char)
        ascii_str = "".join(ascii_parts)
        
        # Print the aligned hex dump line
        print(f"{start_pos + i:08X}  {hex_str:<64} | {ascii_str}")
    print("=" * 60)


def find_and_print_context(filepath, target_byte_int, context_size):
    """Finds the first occurrence of a byte and retrieves the surrounding window."""
    target_byte = bytes([target_byte_int])
    chunk_size = 65536
    offset = 0

    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    print(f"\nTarget byte {hex(target_byte_int)} was NOT found in the file.")
                    break
                
                # Check if our target byte exists in this chunk
                idx = chunk.find(target_byte)
                
                if idx != -1:
                    # Found it! Calculate exact file positions
                    absolute_pos = offset + idx
                    start_pos = max(0, absolute_pos - context_size)
                    
                    # Seek to the start of our context window and read the window size
                    f.seek(start_pos)
                    # Window size is (context before) + (the target itself) + (context after)
                    window = f.read(context_size * 2 + 1)
                    
                    # Determine where the target byte sits inside our isolated window
                    target_pos_in_window = absolute_pos - start_pos
                    
                    display_context(window, target_pos_in_window, absolute_pos, start_pos)
                    break
                
                offset += len(chunk)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Find the first occurrence of a byte and print its context.")
    parser.add_argument("filepath", help="Path to the file to analyze")
    parser.add_argument("target", help="The byte to search for (e.g., 153 or 0x99)")
    parser.add_argument("--context", type=int, default=64, help="Number of bytes before and after to display (default: 64)")
    
    args = parser.parse_args()

    # Parse the target byte. Base 0 allows Python to automatically interpret "0x99" as hex and "153" as decimal.
    try:
        target_byte_int = int(args.target, 0)
        if not (0 <= target_byte_int <= 255):
            raise ValueError
    except ValueError:
        print("Error: Target byte must be an integer between 0 and 255 (e.g., 153 or 0x99).")
        sys.exit(1)

    find_and_print_context(args.filepath, target_byte_int, args.context)

if __name__ == "__main__":
    main()
