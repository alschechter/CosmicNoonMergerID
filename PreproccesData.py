import numpy as np
from glob import glob
from astropy.visualization import simple_norm
from skimage.transform import resize
from SetRandomSeed import set_random_seeds
from tqdm import tqdm
import traceback
import shutil

set_random_seeds(626)
import os
from multiprocessing import Pool, cpu_count

greyscale = 'False'  # options: True (single channel index 2), False (channels [0,1,3]), 'sum' (sum all 3 channels)

def preprocess_single_file(args):
    """Process a single file - must be top-level function for multiprocessing"""
    file, input_path, output_path = args

    try:
        # Load
        image = np.load(file)
        if greyscale == True:
            image = image[:, :, 2]
        elif greyscale == 'sum':
            image = np.sum(image[:, :, [0, 1, 3]], axis=-1)
        else:
            image = image[:, :, [0, 1, 3]]
        
        # Resize
        image = resize(image, (224, 224))
        
        # Asinh normalization
        norm = simple_norm(image, 'asinh', asinh_a=1e-3, clip = True)
        image = norm(image)
        image = np.ma.getdata(image)
        # Clip outliers
        global_min = np.percentile(image, 0.5)
        global_max = np.percentile(image, 99.5)
        image = np.clip(image, a_min=global_min, a_max=global_max)
        
        # Standardize per image
        if (global_max - global_min) > 1e-8:
            image = (image - global_min) / (global_max - global_min)
        else:
            image = image - global_min  # Constant image edge case
        
        if greyscale == True or greyscale == 'sum':
            image = np.stack([image] * 3, axis=-1)
        #print(np.sum(image.mask))
        # Create output path
        rel_path = os.path.relpath(file, input_path)
        output_file = os.path.join(output_path, rel_path)
        
        # Create directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save
        np.save(output_file, image.astype(np.float32))
        
        return True
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return False


# result = preprocess_single_file(
#     ('/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/training/anymergers/allfilters9_6.npy',
#      "/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/",
#      "/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDataset/")
# )



def preprocess_and_save_parallel(input_path, output_path, n_workers=None):
    """Preprocess all images in parallel"""
    
    # Get all files
    files = glob(input_path + '**/*.npy', recursive=True)
    print(f"Found {len(files)} files to process")
    
    # Determine number of workers
    if n_workers is None:
        n_workers = cpu_count() - 1  # Leave one CPU free
    print(f"Using {n_workers} workers")
    
    # Create args for each file
    args_list = [(file, input_path, output_path) for file in files]
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(preprocess_single_file, args_list),
            total=len(args_list),
            desc="Preprocessing"
        ))
    
    successful = sum(results)
    print(f"\nCompleted: {successful}/{len(files)} files processed successfully")

# Run it:

if __name__ == "__main__":
    if greyscale == True:
        preprocess_and_save_parallel(
            input_path='/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/',
            output_path='/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDatasetGrey/',
            n_workers=16  # Or None to auto-detect
        )
    elif greyscale == 'sum':
        preprocess_and_save_parallel(
            input_path='/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/',
            output_path='/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDatasetAllFiltersGrey/',
            n_workers=16  # Or None to auto-detect
        )
    else:
        preprocess_and_save_parallel(
            input_path='/n/holystore01/LABS/hernquist_lab/Users/aschechter/DifferentSplit/',
            output_path='/n/holystore01/LABS/hernquist_lab/Users/aschechter/DifferentSplit/',
            n_workers=16  # Or None to auto-detect
        )





# src_root = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/'
# dst_root = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDatasetAllFiltersGrey/'

# for src in glob(src_root + '**/mergerlabel.npy', recursive=True):
#     dst = src.replace(src_root, dst_root)
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     shutil.copy(src, dst)





# files = glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDataset/**/*.npy', recursive=True)

# print(f"Checking {len(files)} files...")

# corrupted = []
# for file in files:
#     try:
#         img = np.load(file)
#         if img.size == 0 or img.shape != (224, 224, 3):
#             corrupted.append((file, img.shape if img.size > 0 else "EMPTY"))
#     except Exception as e:
#         corrupted.append((file, f"ERROR: {e}"))

# if corrupted:
#     print(f"\nFound {len(corrupted)} corrupted files:")
#     for file, issue in corrupted[:20]:  # Show first 20
#         print(f"  {file}: {issue}")