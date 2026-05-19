import os
import argparse
from pymotion.io.bvh import BVH

def load_bvh_files(directory_path):
    """
    Load all BVH files from the given directory and subdirectories.
    
    Args:
        directory_path: str, path to directory containing BVH files
    
    Returns:
        list: paths to all BVH files found
    """
    bvh_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".bvh"):
                file_path = os.path.join(root, file)
                bvh_files.append(file_path)
    return bvh_files

def count_frames_in_bvh(bvh_path):
    """
    Count the number of frames in a single BVH file.
    
    Args:
        bvh_path: str, path to BVH file
    
    Returns:
        int: number of frames in the BVH file
    """
    try:
        bvh = BVH()
        bvh.load(bvh_path)
        return bvh.data["rotations"].shape[0]
    except Exception as e:
        print(f"Error loading {bvh_path}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Count total frames and calculate running time for BVH database")
    parser.add_argument(
        "database_path",
        type=str,
        help="path to directory containing BVH files"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="frames per second (default: 30.0)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.database_path):
        print(f"Error: Directory {args.database_path} does not exist")
        return
    
    # Load all BVH files from directory
    bvh_files = load_bvh_files(args.database_path)
    
    if not bvh_files:
        print(f"No BVH files found in {args.database_path}")
        return
    
    print(f"Found {len(bvh_files)} BVH files in {args.database_path}")
    print("Processing files...")
    
    total_frames = 0
    processed_files = 0
    
    for bvh_path in bvh_files:
        frames = count_frames_in_bvh(bvh_path)
        if frames > 0:
            total_frames += frames
            processed_files += 1
            print(f"  {os.path.basename(bvh_path)}: {frames} frames")
        else:
            print(f"  {os.path.basename(bvh_path)}: ERROR - could not load")
    
    # Calculate running time
    total_seconds = total_frames / args.fps
    total_minutes = total_seconds / 60.0
    total_hours = total_minutes / 60.0
    
    # Print results
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Total BVH files processed: {processed_files}")
    print(f"Total frames: {total_frames:,}")
    print(f"Frame rate: {args.fps} fps")
    print(f"Total duration: {total_seconds:.2f} seconds")
    print(f"Total duration: {total_minutes:.2f} minutes")
    print(f"Total duration: {total_hours:.2f} hours")
    print("="*50)

if __name__ == "__main__":
    main()