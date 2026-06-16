import os
import shutil
import kagglehub

def main():
    print("Downloading CASIA-Iris-Interval dataset via kagglehub...")
    download_path = kagglehub.dataset_download("swoyam2609/casia-iris-interval")
    print(f"Downloaded to global cache: {download_path}")

    # Target directory in workspace
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    target_dir = os.path.join(repo_root, "datasets", "casia_interval")

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    print(f"Moving dataset files to: {target_dir} ...")
    
    # List contents of the download path
    contents = os.listdir(download_path)
    print(f"Contents of downloaded path: {contents}")
    
    # If the downloaded dataset has a single subdirectory, use that as source
    source_dir = download_path
    if len(contents) == 1 and os.path.isdir(os.path.join(download_path, contents[0])):
        source_dir = os.path.join(download_path, contents[0])
        print(f"Using nested source directory: {source_dir}")
        contents = os.listdir(source_dir)
        print(f"Nested contents: {contents}")

    # Move all items to target directory
    for item in os.listdir(source_dir):
        src_item = os.path.join(source_dir, item)
        dst_item = os.path.join(target_dir, item)
        
        # If target already exists, remove it first to avoid collision
        if os.path.exists(dst_item):
            if os.path.isdir(dst_item):
                shutil.rmtree(dst_item)
            else:
                os.remove(dst_item)
                
        shutil.move(src_item, dst_item)
        print(f"Moved {item} to {target_dir}")

    # Clean up the cache directory for this specific dataset
    dataset_cache_root = os.path.abspath(os.path.join(download_path, "..", ".."))
    if os.path.exists(dataset_cache_root):
        print(f"Removing cache directory to save space: {dataset_cache_root}")
        shutil.rmtree(dataset_cache_root)

    print("Dataset download and setup complete!")

if __name__ == "__main__":
    main()
