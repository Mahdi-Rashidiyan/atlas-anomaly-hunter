"""
ATLAS Open Data Download and Preparation Script
Downloads data from CERN Open Data Portal and prepares it for analysis
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        output_path: Path to save file
        desc: Description for progress bar
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_atlas_data(data_dir: Path, dataset_type: str = 'minimal'):
    """
    Download ATLAS Open Data
    
    Args:
        data_dir: Directory to save data
        dataset_type: 'minimal' (100k events) or 'full' (10/fb)
    """
    print("=" * 80)
    print("ATLAS Open Data Download")
    print("=" * 80)
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for ATLAS Open Data
    base_url = "http://opendata.atlas.cern/release/2020/samples/"
    
    if dataset_type == 'minimal':
        # Download smaller sample for quick testing
        files_to_download = [
            {
                'name': 'data_A.exactly2lep.root',
                'url': base_url + 'data/data_A.exactly2lep.root',
                'size': '~500 MB',
                'description': 'Real collision data (subset)'
            },
            {
                'name': 'mc_410000.ttbar.exactly2lep.root',
                'url': base_url + 'mc/mc_410000.ttbar.exactly2lep.root',
                'size': '~200 MB',
                'description': 'Top quark pair MC simulation'
            }
        ]
    else:
        # Full dataset
        files_to_download = [
            # Real data from different periods
            {'name': 'data_A.exactly2lep.root', 
             'url': base_url + 'data/data_A.exactly2lep.root'},
            {'name': 'data_B.exactly2lep.root', 
             'url': base_url + 'data/data_B.exactly2lep.root'},
            {'name': 'data_C.exactly2lep.root', 
             'url': base_url + 'data/data_C.exactly2lep.root'},
            {'name': 'data_D.exactly2lep.root', 
             'url': base_url + 'data/data_D.exactly2lep.root'},
            
            # Monte Carlo simulations
            {'name': 'mc_410000.ttbar.exactly2lep.root',
             'url': base_url + 'mc/mc_410000.ttbar.exactly2lep.root'},
            {'name': 'mc_361106.Zee.exactly2lep.root',
             'url': base_url + 'mc/mc_361106.Zee.exactly2lep.root'},
            {'name': 'mc_361107.Zmumu.exactly2lep.root',
             'url': base_url + 'mc/mc_361107.Zmumu.exactly2lep.root'},
        ]
    
    print(f"\nDataset type: {dataset_type}")
    print(f"Download directory: {data_dir}")
    print(f"Files to download: {len(files_to_download)}")
    print("\n" + "-" * 80)
    
    for idx, file_info in enumerate(files_to_download, 1):
        filename = file_info['name']
        url = file_info['url']
        output_path = data_dir / filename
        
        print(f"\n[{idx}/{len(files_to_download)}] {filename}")
        if 'description' in file_info:
            print(f"    Description: {file_info['description']}")
        if 'size' in file_info:
            print(f"    Size: {file_info['size']}")
        
        if output_path.exists():
            print(f"    ✓ Already downloaded, skipping...")
            continue
        
        try:
            print(f"    Downloading from: {url}")
            download_file(url, output_path, desc=f"    Progress")
            print(f"    ✓ Download complete!")
        except Exception as e:
            print(f"    ✗ Error downloading {filename}: {str(e)}")
            print(f"    You can manually download from: {url}")
            continue
    
    print("\n" + "=" * 80)
    print("Download complete!")
    print("=" * 80)


def verify_data(data_dir: Path):
    """
    Verify downloaded data files
    
    Args:
        data_dir: Data directory to check
    """
    print("\nVerifying data files...")
    print("-" * 80)
    
    root_files = list(data_dir.glob("*.root"))
    
    if not root_files:
        print("⚠️  No ROOT files found in data directory!")
        print(f"   Check: {data_dir}")
        return False
    
    print(f"✓ Found {len(root_files)} ROOT files:")
    
    total_size = 0
    for root_file in root_files:
        size_mb = root_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  • {root_file.name} ({size_mb:.1f} MB)")
    
    print(f"\nTotal size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print("-" * 80)
    
    # Try to open files with uproot
    try:
        import uproot
        print("\nTesting file accessibility with uproot...")
        
        for root_file in root_files[:2]:  # Test first 2 files
            try:
                file = uproot.open(root_file)
                tree = file["mini"]
                n_events = tree.num_entries
                print(f"  ✓ {root_file.name}: {n_events:,} events")
            except Exception as e:
                print(f"  ✗ {root_file.name}: Error - {str(e)}")
        
        print("\n✓ Data verification complete!")
        return True
        
    except ImportError:
        print("⚠️  uproot not installed - skipping detailed verification")
        print("   Install with: pip install uproot awkward")
        return True


def create_sample_dataset(data_dir: Path, output_file: str = "sample_events.csv", n_events: int = 1000):
    """
    Create a small CSV sample for quick testing
    
    Args:
        data_dir: Data directory
        output_file: Output CSV filename
        n_events: Number of events to extract
    """
    try:
        import uproot
        import pandas as pd
        from atlas_anomaly_detector import ATLASDataLoader
        
        print(f"\nCreating sample dataset ({n_events} events)...")
        
        # Find first data file
        root_files = list(data_dir.glob("data_*.root"))
        if not root_files:
            root_files = list(data_dir.glob("*.root"))
        
        if not root_files:
            print("No ROOT files found!")
            return
        
        # Load and process
        loader = ATLASDataLoader(data_dir)
        data = loader.prepare_data([str(root_files[0])], max_events_per_file=n_events)
        
        # Save to CSV
        output_path = data_dir / output_file
        data.to_csv(output_path, index=False)
        
        print(f"✓ Sample dataset saved to: {output_path}")
        print(f"  Events: {len(data)}")
        print(f"  Features: {len(data.columns)}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        
    except ImportError as e:
        print(f"⚠️  Cannot create sample: {str(e)}")
        print("   Install required packages: pip install uproot awkward pandas")
    except Exception as e:
        print(f"✗ Error creating sample: {str(e)}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Download and prepare ATLAS Open Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download minimal dataset (recommended for testing)
  python download_atlas_data.py --type minimal
  
  # Download full dataset
  python download_atlas_data.py --type full --data-dir ./data
  
  # Only verify existing data
  python download_atlas_data.py --verify-only
  
  # Create sample CSV
  python download_atlas_data.py --create-sample --sample-size 5000
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to store data (default: ./data)'
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['minimal', 'full'],
        default='minimal',
        help='Dataset type: minimal (2 files, ~700MB) or full (7+ files, ~5GB)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data, skip download'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample CSV file for quick testing'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,
        help='Number of events in sample CSV (default: 1000)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Print header
    print("\n" + "=" * 80)
    print("ATLAS Open Data - Download and Preparation Tool")
    print("=" * 80)
    print(f"\nData directory: {data_dir.absolute()}")
    
    # Download data (unless verify-only)
    if not args.verify_only:
        print(f"\nDownloading {args.type} dataset...")
        download_atlas_data(data_dir, args.type)
    
    # Verify data
    if data_dir.exists():
        verify_success = verify_data(data_dir)
        
        # Create sample if requested
        if args.create_sample and verify_success:
            create_sample_dataset(data_dir, n_events=args.sample_size)
    else:
        print(f"\n⚠️  Data directory does not exist: {data_dir}")
    
    # Final instructions
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Verify data was downloaded successfully")
    print("2. Run the main analysis:")
    print("   python atlas_anomaly_detector.py")
    print("\n3. Or explore interactively:")
    print("   jupyter notebook atlas_anomaly_analysis.ipynb")
    print("\n4. For real-time detection demo:")
    print("   python real_time_detector.py")
    print("\n" + "=" * 80)
    print("\n📚 Documentation:")
    print("   • README.md - Project overview")
    print("   • ATLAS Open Data: http://opendata.atlas.cern")
    print("   • CERN openlab: https://openlab.cern")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()