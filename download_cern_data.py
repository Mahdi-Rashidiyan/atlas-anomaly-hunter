"""
ATLAS Data Downloader - Direct from CERN Open Data
Downloads ROOT files for ATLAS 13 TeV 2-lepton analysis
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm
import json


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading"):
    """Download file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        try:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
            return True
        except Exception as e:
            print(f"    Error: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            return False


def get_cern_file_list():
    """Get list of available files from CERN record 15007"""
    # These are the actual data files from CERN record 15007
    files = [
        # Real data files
        ("data_A.exactly2lep.root", "http://opendata.cern.ch/eos/opendata/atlas/AtlasOpenData2020/rucio/atlas/atlasdata/data_A.exactly2lep.root"),
        ("data_B.exactly2lep.root", "http://opendata.cern.ch/eos/opendata/atlas/AtlasOpenData2020/rucio/atlas/atlasdata/data_B.exactly2lep.root"),
        ("data_C.exactly2lep.root", "http://opendata.cern.ch/eos/opendata/atlas/AtlasOpenData2020/rucio/atlas/atlasdata/data_C.exactly2lep.root"),
        
        # MC samples (most important for anomaly detection)
        ("mc_301215.ZPrime2000_ee.exactly2lep.root", "http://opendata.cern.ch/eos/opendata/atlas/AtlasOpenData2020/rucio/atlas/Zp_2000_ee/mc_301215.ZPrime2000_ee.exactly2lep.root"),
        ("mc_410000.ttbar.exactly2lep.root", "http://opendata.cern.ch/eos/opendata/atlas/AtlasOpenData2020/rucio/atlas/ttbar/mc_410000.ttbar.exactly2lep.root"),
        ("mc_361106.Zee.exactly2lep.root", "http://opendata.cern.ch/eos/opendata/atlas/AtlasOpenData2020/rucio/atlas/Zee/mc_361106.Zee.exactly2lep.root"),
        ("mc_361107.Zmumu.exactly2lep.root", "http://opendata.cern.ch/eos/opendata/atlas/AtlasOpenData2020/rucio/atlas/Zmumu/mc_361107.Zmumu.exactly2lep.root"),
    ]
    return files


def main():
    print("\n" + "=" * 80)
    print("ATLAS Open Data Download - Record 15007")
    print("2-Lepton Event Analysis")
    print("=" * 80)
    
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    files = get_cern_file_list()
    
    print(f"\nTarget directory: {data_dir.absolute()}")
    print(f"Files to download: {len(files)}")
    print(f"Approximate size: 5-10 GB\n")
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for idx, (filename, url) in enumerate(files, 1):
        output_path = data_dir / filename
        
        print(f"\n[{idx}/{len(files)}] {filename}")
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ Already exists ({size_mb:.1f} MB), skipping")
            skipped += 1
            continue
        
        print(f"    URL: {url}")
        success = download_file(url, output_path, desc=f"    Downloading")
        
        if success:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ Downloaded ({size_mb:.1f} MB)")
            downloaded += 1
        else:
            print(f"    ✗ Failed to download")
            failed += 1
    
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Downloaded:  {downloaded}/{len(files)}")
    print(f"Skipped:     {skipped}/{len(files)}")
    print(f"Failed:      {failed}/{len(files)}")
    
    # List downloaded files
    root_files = list(data_dir.glob("*.root"))
    if root_files:
        total_size = sum(f.stat().st_size for f in root_files) / (1024 * 1024 * 1024)
        print(f"\nDownloaded ROOT files: {len(root_files)}")
        print(f"Total size: {total_size:.2f} GB")
        print("\nFiles:")
        for f in sorted(root_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 80)
    print("NEXT: Run anomaly detection")
    print("=" * 80)
    print("\nTo process the downloaded data:")
    print("  python process_atlas_data.py\n")


if __name__ == "__main__":
    main()
