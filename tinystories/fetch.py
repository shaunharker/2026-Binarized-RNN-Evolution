import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    # We use the '-hf' variant because the original 'roneneldan/TinyStories' 
    # dataset loader splits rows by newlines rather than by full stories.
    dataset_name = "skeskinen/TinyStories-hf"
    output_filename = "tinystories_all.txt"

    print(f"Loading '{dataset_name}' from Hugging Face...")
    # Load the dataset (this downloads and caches it locally)
    dataset = load_dataset(dataset_name)

    # Calculate total stories for informational purposes
    total_stories = sum(len(dataset[split]) for split in dataset.keys())
    print(f"Total stories found: {total_stories:,}")

    # Open the text file in write mode
    with open(output_filename, "w", encoding="utf-8") as outfile:
        
        # Iterate over available splits (usually 'train' and 'validation')
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\nProcessing '{split_name}' split ({len(split_data):,} stories)...")
            
            # Use tqdm to show a progress bar
            for row in tqdm(split_data, desc=f"Writing {split_name}", unit="story"):
                story_text = row["text"].strip()
                
                # Write the story followed by a delimiter
                # You can change <|endoftext|> to "\n\n" if you just want blank lines
                outfile.write(story_text)
                outfile.write("\n\n")

    print(f"\nDone! All stories have been concatenated and saved to '{output_filename}'.")

if __name__ == "__main__":
    main()
