#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Benchmark Downloader

This script downloads various video benchmark datasets to a specified directory.
It supports the following benchmarks:
- Perception Test
- TVBench
- VinoGround
- TempCompass
- CinePile
- MSRVTT
- NextQA
"""

import argparse
import os
import sys
import requests
import importlib
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['tqdm', 'huggingface_hub', 'gitpython']
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("Missing required packages. Please install them using:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    # Now import required modules
    global tqdm, hf_hub_download, git
    from tqdm import tqdm
    from huggingface_hub import hf_hub_download
    import git

def create_parser():
    """Create argument parser for the script."""
    parser = argparse.ArgumentParser(description='Download video benchmark datasets.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save the benchmark datasets')
    parser.add_argument('--benchmarks', nargs='+', 
                        choices=['perception_test', 'tvbench', 'vinoground', 
                                'tempcompass', 'cinepile', 'msrvtt', 'nextqa', 'all'],
                        default=['all'], 
                        help='Benchmarks to download')
    return parser

def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Created directory: {dir_path}")

def download_file(url, save_path, chunk_size=8192):
    """
    Download a file with progress bar.
    
    Args:
        url (str): URL to download
        save_path (str): Path to save the file
        chunk_size (int): Size of chunks to download
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        logger.info(f"Downloaded to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def clone_repository(git_url, target_dir, depth=1):
    """
    Clone a git repository.
    
    Args:
        git_url (str): Git repository URL
        target_dir (str): Directory to clone into
        depth (int): Git clone depth (1 for shallow clone)
        
    Returns:
        bool: True if clone successful, False otherwise
    """
    try:
        logger.info(f"Cloning repository from {git_url}...")
        git.Repo.clone_from(git_url, target_dir, depth=depth)
        logger.info(f"Repository cloned to {target_dir}")
        return True
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        return False

def download_perception_test(output_dir):
    """Download Perception Test benchmark."""
    try:
        perception_dir = os.path.join(output_dir, 'perception_test')
        create_directory(perception_dir)
        
        # Clone the repository
        git_url = "https://github.com/deepmind/perception_test.git"
        return clone_repository(git_url, perception_dir)
    except Exception as e:
        logger.error(f"Error downloading Perception Test: {e}")
        return False

def download_tvbench(output_dir):
    """Download TVBench benchmark."""
    try:
        tvbench_dir = os.path.join(output_dir, 'tvbench')
        create_directory(tvbench_dir)
        
        # Clone the repository
        git_url = "https://github.com/daniel-cores/tvbench.git"
        clone_success = clone_repository(git_url, tvbench_dir)
        
        if clone_success:
            # Create a script to download the dataset
            script_path = os.path.join(tvbench_dir, 'download_tvbench.py')
            with open(script_path, 'w') as f:
                f.write('#!/usr/bin/env python3\n')
                f.write('from datasets import load_dataset\n\n')
                f.write('print("Downloading TVBench dataset...")\n')
                f.write('dataset = load_dataset("daniel-cores/tvbench")\n')
                f.write('print(f"Dataset downloaded successfully. Available splits: {list(dataset.keys())}")\n')
            
            os.chmod(script_path, 0o755)
            logger.info(f"Created download script at {script_path}")
        
        return clone_success
    except Exception as e:
        logger.error(f"Error downloading TVBench: {e}")
        return False

def download_vinoground(output_dir):
    """Download Vinoground benchmark."""
    try:
        vinoground_dir = os.path.join(output_dir, 'vinoground')
        create_directory(vinoground_dir)
        
        # Clone the repository
        git_url = "https://github.com/Vinoground/Vinoground.git"
        clone_success = clone_repository(git_url, vinoground_dir)
        
        if clone_success:
            # Create instructions for downloading from HuggingFace
            with open(os.path.join(vinoground_dir, 'download_instructions.md'), 'w') as f:
                f.write("# Vinoground Dataset Download Instructions\n\n")
                f.write("To download the complete dataset from HuggingFace:\n\n")
                f.write("```
                f.write("cd Vinoground\n")
                f.write("git clone https://huggingface.co/datasets/Vinoground/Vinoground .\n")
                f.write("unzip vinoground_videos.zip\n")
                f.write("unzip vinoground_videos_concated.zip\n")
                f.write("```\n")
        
        return clone_success
    except Exception as e:
        logger.error(f"Error downloading Vinoground: {e}")
        return False

def download_tempcompass(output_dir):
    """Download TempCompass benchmark."""
    try:
        tempcompass_dir = os.path.join(output_dir, 'tempcompass')
        create_directory(tempcompass_dir)
        
        # Create a README with instructions since no clear download URL is available
        readme_path = os.path.join(tempcompass_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write("# TempCompass Benchmark\n\n")
            f.write("TempCompass is a diagnostic benchmark for evaluating temporal perception abilities of Video LLMs.\n\n")
            f.write("Please refer to the original paper for more information on how to access the dataset.\n")
            f.write("Paper reference: \"Do Video LLMs Really Understand Videos?\"\n")
        
        logger.info(f"Created TempCompass directory at {tempcompass_dir}")
        logger.info("Note: No direct download link available for TempCompass.")
        return True
    except Exception as e:
        logger.error(f"Error creating TempCompass directory: {e}")
        return False

def download_cinepile(output_dir):
    """Download CinePile benchmark."""
    try:
        cinepile_dir = os.path.join(output_dir, 'cinepile')
        create_directory(cinepile_dir)
        
        # Create a script to download the dataset using Hugging Face
        script_path = os.path.join(cinepile_dir, 'download_cinepile.py')
        with open(script_path, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('from datasets import load_dataset\n\n')
            f.write('print("Downloading CinePile dataset...")\n')
            f.write('dataset = load_dataset("tomg-group-umd/cinepile")\n\n')
            f.write('print(f"Dataset downloaded successfully.")\n')
            f.write('print(f"Train split: {len(dataset[\"train\"])} samples")\n')
            f.write('if "test" in dataset:\n')
            f.write('    print(f"Test split: {len(dataset[\"test\"])} samples")\n')
        
        os.chmod(script_path, 0o755)
        logger.info(f"Created download script at {script_path}")
        return True
    except Exception as e:
        logger.error(f"Error setting up CinePile: {e}")
        return False

def download_msrvtt(output_dir):
    """Download MSRVTT benchmark."""
    try:
        msrvtt_dir = os.path.join(output_dir, 'msrvtt')
        create_directory(msrvtt_dir)
        
        # Clone MMAction2 repository to get download scripts
        git_url = "https://github.com/open-mmlab/mmaction2.git"
        clone_success = clone_repository(git_url, os.path.join(msrvtt_dir, 'mmaction2'))
        
        if clone_success:
            # Create a convenience script to run the MSRVTT download
            script_path = os.path.join(msrvtt_dir, 'setup_msrvtt.sh')
            with open(script_path, 'w') as f:
                f.write('#!/bin/bash\n\n')
                f.write('cd mmaction2/tools/data/msrvtt\n')
                f.write('echo "Downloading MSRVTT videos..."\n')
                f.write('bash download_msrvtt.sh\n')
                f.write('echo "Preprocessing videos..."\n')
                f.write('bash compress_msrvtt.sh\n')
                f.write('echo "MSRVTT setup completed."\n')
            
            os.chmod(script_path, 0o755)
            logger.info(f"Created setup script at {script_path}")
        
        return clone_success
    except Exception as e:
        logger.error(f"Error downloading MSRVTT: {e}")
        return False

def download_nextqa(output_dir):
    """Download NextQA benchmark."""
    try:
        nextqa_dir = os.path.join(output_dir, 'nextqa')
        create_directory(nextqa_dir)
        
        # Clone the repository
        git_url = "https://github.com/doc-doc/NExT-QA.git"
        return clone_repository(git_url, nextqa_dir)
    except Exception as e:
        logger.error(f"Error downloading NextQA: {e}")
        return False

def main():
    """Main function to download benchmarks."""
    # Check dependencies first
    check_dependencies()
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create the output directory
    create_directory(args.output_dir)
    
    # Determine which benchmarks to download
    benchmarks_to_download = args.benchmarks
    if 'all' in benchmarks_to_download:
        benchmarks_to_download = ['perception_test', 'tvbench', 'vinoground', 
                                  'tempcompass', 'cinepile', 'msrvtt', 'nextqa']
    
    # Keep track of results
    results = {}
    
    # Download each benchmark
    for benchmark in benchmarks_to_download:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Downloading {benchmark}...")
        logger.info(f"{'=' * 50}")
        
        start_time = time.time()
        
        if benchmark == 'perception_test':
            results[benchmark] = download_perception_test(args.output_dir)
        elif benchmark == 'tvbench':
            results[benchmark] = download_tvbench(args.output_dir)
        elif benchmark == 'vinoground':
            results[benchmark] = download_vinoground(args.output_dir)
        elif benchmark == 'tempcompass':
            results[benchmark] = download_tempcompass(args.output_dir)
        elif benchmark == 'cinepile':
            results[benchmark] = download_cinepile(args.output_dir)
        elif benchmark == 'msrvtt':
            results[benchmark] = download_msrvtt(args.output_dir)
        elif benchmark == 'nextqa':
            results[benchmark] = download_nextqa(args.output_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Download Summary")
    logger.info("=" * 50)
    
    for benchmark, success in results.items():
        status = "Success" if success else "Failed"
        logger.info(f"{benchmark}: {status}")
    
    logger.info("\nNote: Some benchmarks may require additional steps to complete the download.")
    logger.info("Please check the README or instruction files in each directory.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)
