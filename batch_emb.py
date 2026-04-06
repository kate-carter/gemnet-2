# API Script for generating an entire set (all embeddings at all
# dimensionalities) of data at a time
# generates and saves 3072 embeddings and then generates
# 1536, which is then cropped to 768, 512, 256 and normed AFTER cropping

#imports
import os #does not need to be installed just imported
from typing import List, Optional
import numpy as np
import google
from google import genai
from google.genai import types
model = "gemini-embedding-001"

# List of files to process
input_files = [
    'banana_w.rtf',
    'bottle_w.rtf',
    'car_w.rtf',
    'chocolate_w.rtf',
    'cookie_w.rtf',
    'grass_w.rtf',
    'key_w.rtf',
    'knife_w.rtf',
    'paper_w.rtf',
    'phone_w.rtf',
    'pickle_w.rtf',
    'pillow_w.rtf',
    'pumpkin_w.rtf',
    'table_w.rtf',
    'wood_w.rtf'
]

# Base paths
input_dir = '/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/old_pipeline/take3/concepts_wterm/'
output_dir = '/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/embeddings-001/'

#Dimensionality, model, and task options
#task type is SEMANTIC_SIMILARITY automatically
#define embed_text
def embed_texts(client, texts, dim):
    return client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=dim
        ),
    )


# --- Environment Setup ---
def setup_environment():
    """Prompts user to set the GEMINI_API_KEY environment variable if not already set."""
    if "GOOGLE_API_KEY" not in os.environ:
        print("=" * 60)
        print("GEMINI API KEY SETUP REQUIRED")
        print("=" * 60)
        print("This script requires a Gemini API key to generate embeddings.")
        print("=" * 60)
        return False
    return True


# read RTF files because sometimes syntax is weird
def read_rtf_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if content.startswith('{\\rtf'):
            import re

            # Step 1: Remove font table ({\fonttbl...})
            content = re.sub(r'\{\\fonttbl[^}]*\}', '', content)

            # Step 2: Remove color table ({\colortbl...})
            content = re.sub(r'\{\\colortbl[^}]*\}', '', content)

            # Step 3: Remove stylesheet ({\stylesheet...})
            content = re.sub(r'\{\\stylesheet[^}]*\}', '', content)

            # Step 4: Remove info group ({\info...})
            content = re.sub(r'\{\\info[^}]*\}', '', content)

            # Step 5: Remove other groups like {\*\...}
            content = re.sub(r'\{\\\*[^}]*\}', '', content)

            # Step 6: Remove RTF control words with parameters
            content = re.sub(r'\\[a-z]+\-?\d+\s?', ' ', content)

            # Step 7: Remove RTF control words without parameters
            content = re.sub(r'\\[a-z]+\s?', ' ', content)

            # Step 8: Remove special characters like \*, \', etc.
            content = re.sub(r'\\[^a-z\s]', '', content)

            # Step 9: Remove remaining braces
            content = re.sub(r'[{}]', '', content)

            # Step 10: Clean up multiple spaces and semicolons
            content = re.sub(r'\s*;\s*', '', content)  # Remove semicolons
            content = re.sub(r'\s+', ' ', content)  # Collapse whitespace
            content = content.strip()

            # Step 11: Ensure exactly 50 comma separated values
            values = content.split(',')
            if len(values) != 50:
                print(f"Warning: Expected 50 values, got {len(values)}")
                return None
            content = ','.join(values[:50])

        return content
    except Exception as e:
        print(f"Error reading RTF file {file_path}: {e}")
        return None


if __name__ == "__main__":
    from datetime import datetime
    import csv
        
    # Configure the API key
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    # 1. Check for API Key and prompt user if needed
    if not setup_environment():
        print("\nScript terminated. Please set your API key and run again.")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each input file
    for file_idx, input_filename in enumerate(input_files, 1):
        print(f"\n{'=' * 70}")
        print(f"Processing file {file_idx}/{len(input_files)}: {input_filename}")
        print(f"{'=' * 70}")

        # Construct full input path
        input_path = os.path.join(input_dir, input_filename)

        # Extract concept name from filename (e.g., "banana_csv.rtf" -> "banana")
        concept_name = input_filename.replace('csv.rtf', '').replace('_w.rtf', '')

        # Read the RTF file
        linguistics_text = read_rtf_file(input_path)

        if not linguistics_text:
            print(f"✗ Failed to read file: {input_filename}")
            continue

        print(f"Successfully read file. Content length: {len(linguistics_text)} characters")
        print(f"Number of values: {len(linguistics_text.split(','))}")
        print(f"First 100 characters: {linguistics_text[:50]}...")

        # Parse comma-separated texts
        texts_to_analyze = [text.strip() for text in linguistics_text.split(',') if text.strip()]
        print(f"\nAnalyzing {len(texts_to_analyze)} texts from {concept_name}")

        # Generate embeddings for each text
        all_embeddings = {}
        failed_texts = []
        import time

        #define target dimensionality
        dims_to_generate = (3072, 1536)
        embeddings_by_dim = {dim: {} for dim in dims_to_generate}
        failed_by_dim = {dim: [] for dim in dims_to_generate}

        #loop to generate both size of embedding
        for dim in dims_to_generate:
            try:
                response = embed_texts(client, texts_to_analyze, dim)
            except Exception as e:
                failed_by_dim[dim].extend(texts_to_analyze)
                print(f"✗ Failed to generate embeddings at dim={dim} for {input_filename}")
                continue

            #unpack into dict keyed by text
            for text, embedding in zip(texts_to_analyze, response.embeddings):
                embeddings_by_dim[dim][text] = embedding.values
            
        for dim in dims_to_generate:
            if len(embeddings_by_dim[dim]) == 0:
                print(f"✗ No embeddings generated at dim={dim} for {input_filename}")
            elif failed_by_dim[dim]:
                print(f"⚠ Some texts failed at dim={dim}: {failed_by_dim[dim]}")
                continue
        
        total_successes = sum(len(d) for d in embeddings_by_dim.values())

        if total_successes == 0:
                print(f"\n ✗ Failed to generate embeddings for any text in {input_filename}")
                
        if len(embeddings_by_dim[3072]) > 0:

            # Write CSV headers with text names as columns
            output_filename = f"{concept_name}_3072.csv"
            output_path = os.path.join(output_dir, output_filename)
            headers = ["Dimension"] + list(embeddings_by_dim[3072].keys())
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # metadata
                writer.writerow(["Multi-Text Embedding Vector Information"])
                writer.writerow(["Source file", input_filename])
                writer.writerow(["Concept", concept_name])
                writer.writerow(["Texts analyzed", ", ".join(list(embeddings_by_dim[3072].keys()))])
                writer.writerow(["Model", model])
                writer.writerow(["Dimension per text", 3072])
                writer.writerow(["Generated on", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([])
                
                #write column headers
                writer.writerow(headers)

                #write all vector dimensions as rows
                dimension_count = len(list(embeddings_by_dim[3072].values())[0])
                for dim_idx in range(dimension_count):
                    row = [f"Dimension_{dim_idx + 1}"]
                    for text in embeddings_by_dim[3072].keys():
                        row.append(embeddings_by_dim[3072][text][dim_idx])
                    writer.writerow(row)

            print(f"\n✓ Successfully created CSV file:")
            print(f"  Output: {output_path}")
            print(f"  Texts processed: {len(embeddings_by_dim[3072])}")
            print(f"  Dimensions per text: {dimension_count}")

                
            if len(embeddings_by_dim[1536]) > 0:

                #crop 1536's to 768, 512, 256 separate matrices
                crop_dims = (1536,768, 512, 256)
                for dim in crop_dims:
                            
                    #crop the embedding to the desired dimension
                    texts = list(embeddings_by_dim[1536].keys())
                    embedding_1536 = np.array([embeddings_by_dim[1536][t] for t in texts], dtype=float)
                    embedding_cropped = embedding_1536[:, :dim]
                            
                    #2-norm them all
                    norms = np.linalg.norm(embedding_cropped, axis=1, keepdims=True)
                    normed = embedding_cropped / norms
                    print(f"cropped embedding {dim} normed successfully") 

                    output_filename = f"{concept_name}_{dim}.csv"
                    output_path = os.path.join(output_dir, output_filename)

                    #save csv file with correct dimensionality
                    with open(output_path, 'w', newline='') as f:
                        writer = csv.writer(f)

                        # headers (same as the 3072 case)
                        writer.writerow(["Multi-Text Embedding Vector Information"])
                        writer.writerow(["Source file", input_filename])
                        writer.writerow(["Concept", concept_name])
                        writer.writerow(["Texts analyzed", ", ".join(list(embeddings_by_dim[1536].keys()))])
                        writer.writerow(["Model", model])
                        writer.writerow(["Dimension per text", dim])
                        writer.writerow(["Generated on", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                        writer.writerow([])

                        # write column headers and rows from "normed"
                        num_texts = len(texts)
                        writer.writerow(["Dimension"] + texts)

                        #write rows
                        for dim_idx in range(dim):
                            row = [f"Dimension_{dim_idx + 1}"]
                            for text_idx in range(num_texts):
                                row.append(normed[text_idx, dim_idx])
                            writer.writerow(row)

                        print(f"✓ Successfully created CSV file:")
                        print(f"  Output: {output_path}")
                        print(f"  Texts processed: {num_texts}")
                        print(f"  Dimensions per text: {dim}")

            else:
                print(f"\n✓ Successfully generated embeddings for {total_successes} texts in {input_filename}")

    print(f"\n{'=' * 70}")
    print(f"Processing complete! Processed {len(input_files)} files.")
    print(f"{'=' * 70}")