
import os
import torch
import glob
import sys
import argparse

# Add src to path to import dataset modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset.caching import get_cache_path

def count_lines_in_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def extract_stats(data_dir, dataset_type, txt_file=None):
    print(f"\n{'='*20} Analyzing {dataset_type.upper()} SET {'='*20}")
    
    # 1. Count from Text File (Source of Truth for "Original Radargrams")
    if txt_file and os.path.exists(txt_file):
        num_lines = count_lines_in_file(txt_file)
        # Assuming 2 lines per pair (one rgram, one sim)
        estimated_pairs = num_lines // 2
        print(f"Source File: {os.path.basename(txt_file)}")
        print(f"  Total Lines: {num_lines}")
        print(f"  Estimated Pairs (Lines / 2): {estimated_pairs}")
    elif txt_file:
         print(f"Warning: Source file {txt_file} not found.")

    # 2. Count Files on Disk (Valid Downloaded Inputs)
    real_dir = os.path.join(data_dir, dataset_type, "real")
    sim_dir = os.path.join(data_dir, dataset_type, "sim")
    
    num_real = len(glob.glob(os.path.join(real_dir, "*.xml")))
    num_sim = len(glob.glob(os.path.join(sim_dir, "*.xml")))
    
    print(f"Files on Disk:")
    print(f"  Real XMLs: {num_real}")
    print(f"  Sim XMLs:  {num_sim}")
    
    # 3. Analyze Processed Dataset (.pt file)
    # We need to reconstruct the cache path based on a default config or look for any .pt file
    # For now, let's look for *any* .pt file in the root of the dataset folder 
    # (since the cache hash depends on config which we might not fully know here without loading config.yaml)
    
    pt_files = glob.glob(os.path.join(data_dir, dataset_type, "*.pt"))
    if not pt_files:
        print("  Processed Dataset (.pt): NOT FOUND")
    else:
        for pt_file in pt_files:
            try:
                print(f"Processed Dataset ({os.path.basename(pt_file)}):")
                data = torch.load(pt_file)
                
                # Check for stats dict
                stats = data.get('stats')
                if stats:
                    print(f"  [Cached Stats] Valid Pairs: {stats.get('valid_pairs', 'N/A')}")
                    print(f"  [Cached Stats] Total Patches: {stats.get('total_patches', 'N/A')}")
                
                # Count patches directly
                real_data = data.get('real')
                if real_data is not None:
                    if isinstance(real_data, list):
                        total_patches = sum(t.shape[0] for t in real_data)
                    else:
                        total_patches = len(real_data)
                    print(f"  Actual Patches in Tensor: {total_patches}")
                
                config = data.get('config')
                if config:
                    print(f"  Patch Size: {config.get('patch_size')}")
                    print(f"  Overlap: {config.get('patch_overlap')}")
                    
            except Exception as e:
                print(f"  Error reading {pt_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Dataset Statistics")
    parser.add_argument("--root", type=str, default="/media/datapart/samueletrainotti/data", help="Root data directory")
    args = parser.parse_args()
    
    # Hardcoded based on user info
    # TRAIN
    extract_stats(args.root, "train", os.path.join(args.root, "elysium_planitia.txt"))
    
    # VAL
    extract_stats(args.root, "val", os.path.join(args.root, "url_val.txt"))
    
    # TEST - assuming "test" folder exists or is "underground_structures" mapped to test?
    # User said underground_structures.txt is for testing.
    # Usually test data might be in `data/test`.
    if os.path.exists(os.path.join(args.root, "test")):
         extract_stats(args.root, "test", os.path.join(args.root, "underground_structures.txt"))
    else:
         print(f"\n{'='*20} Analyzing TEST SET {'='*20}")
         print("Validation/Test folder 'test' not found in root. Checking if it's named differently or processed elsewhere.")
         # Check if underground_structures matches a folder
