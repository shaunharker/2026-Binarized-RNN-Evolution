# training_data

This folder contains some scripts for fetching, cleaning/converting, and briefly analyzing textual training data.

All that is really required for GA or QAT to run is an ASCII text file. If you don't have one, you can run through the pipeline here to fetch the "tinystories" corpus and convert it to ascii. Alternatively, if you have your own text file but it isn't ASCII, you can use the ascii converter (does a few transformations).

## Fetching data

### `fetch.py`

downloads a huggingface dataset "tinystories"

## Cleaning/Converting Data

I ran into trouble with tinystories since it had more than 128 unique characters, disrupting my design with 128 token ids. The following three scripts dealt with that.

### `unique.py`

counts each unique byte that occurs. this led to the discovery of chinese characters in the tinystories dataset, for instance.

### `byte_context.py`

retrieve excerpt around first instance of a given byte. I used this after finding unexpected bytes with `unique.py` in order to figure out the context of the offending byte. apparently it was rare Chinese output

### `clean_to_ascii.py`

cleans up non-ascii text, such as the Chinese stories found in the tinystories dataset.

## Analyzing Data

Beyond simple filesize and word count, we can look at entropy.

### `entropy.py`

gives n-gram entropy statistics. we can compare these figures to cross-entropy achieved by the BRNN
