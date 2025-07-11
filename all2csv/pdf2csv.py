#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract tables from PDF files using camelot and export as CSV files.

Requirements:
    pip install camelot-py[cv] pandas
    
System dependencies:
    - ghostscript (for PDF processing)
    - Ubuntu/Debian: sudo apt-get install ghostscript python3-tk
    - macOS: brew install ghostscript
"""

import os
import sys
from pathlib import Path
import pandas as pd
import camelot

def extract_pdf_tables_to_csv(
    pdf_path: str,
    output_dir: str = "pdf",
    pages: str = "all",
    flavor: str = "stream",
) -> list[Path]:
    """
    Extract all tables from a PDF file and save each as a CSV.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save CSV files (default: 'pdf').
        pages: Pages to extract from. Can be 'all', '1', '1,2,3', '1-3', etc.
        flavor: Parsing flavor - 'stream' (default) or 'lattice'.
               - 'stream': Better for tables without clear borders
               - 'lattice': Better for tables with clear borders

    Returns:
        List of generated CSV Path objects.
    """
    pdf_path = Path(pdf_path)
    
    # Check if PDF file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory if it doesn't exist
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting tables from: {pdf_path}")
    print(f"Output directory: {out_dir}")
    print(f"Pages: {pages}")
    print(f"Flavor: {flavor}")
    print("-" * 50)
    
    try:
        # Extract tables using camelot
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=pages,
            flavor=flavor,
        )
        
        print(f"Found {len(tables)} tables")
        
        if len(tables) == 0:
            print("No tables found in the PDF.")
            print(" Tips:")
            print("   - Try using flavor='lattice' instead of 'stream'")
            print("   - Check if the PDF contains actual tables (not just images)")
            print("   - Verify the page numbers are correct")
            return []
        
        csv_paths = []
        
        for i, table in enumerate(tables, 1):
            # Generate filename
            csv_filename = f"table_{i}_page{table.page}.csv"
            csv_path = out_dir / csv_filename
            
            # Save table as CSV
            table.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
            csv_paths.append(csv_path)
            
            # Print table info
            print(f"Table {i} (Page {table.page}):")
            print(f"   - Shape: {table.shape} (rows x columns)")
            print(f"   - Accuracy: {table.accuracy:.2f}%")
            print(f"   - Saved to: {csv_path}")
            print()
        
        return csv_paths
        
    except Exception as e:
        print(f" Error extracting tables: {e}")
        print("\n Troubleshooting:")
        print("1. Make sure ghostscript is installed on your system")
        print("2. Try different 'flavor' parameter ('lattice' or 'stream')")
        print("3. Check if the PDF is not password protected")
        print("4. Verify the PDF contains actual tables (not just images of tables)")
        return []


def test_single_page(pdf_path: str, page_num: int = 12):
    """Test extraction on a single page first."""
    print(f" Testing extraction on page {page_num}...")
    
    csv_files = extract_pdf_tables_to_csv(
        pdf_path=pdf_path,
        output_dir="pdf",
        pages=str(page_num),
        flavor="stream"
    )
    
    if csv_files:
        print(f" Successfully extracted {len(csv_files)} tables from page {page_num}")
        return True
    else:
        print(f" No tables found on page {page_num}")
        # Try with lattice flavor
        print(" Trying with 'lattice' flavor...")
        csv_files = extract_pdf_tables_to_csv(
            pdf_path=pdf_path,
            output_dir="pdf",
            pages=str(page_num),
            flavor="lattice"
        )
        
        if csv_files:
            print(f"Successfully extracted {len(csv_files)} tables from page {page_num} using lattice flavor")
            return True
        else:
            print(f" Still no tables found on page {page_num}")
            return False


def main():
    """Main function to run the PDF table extraction."""
    
    # Configuration
    pdf_path = "tsmc_2024_yearly report.pdf"
    test_page = 12  # First page with tables
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f" PDF file not found: {pdf_path}")
        print("Available PDF files in current directory:")
        for file in Path(".").glob("*.pdf"):
            print(f"   - {file}")
        return
    
    print(" PDF Table Extraction Tool")
    print("=" * 50)
    
    # Test extraction on single page first
    success = test_single_page(pdf_path, test_page)
    
    if success:
        print("\n" + "=" * 50)
        print(" Test successful! Now extracting from all pages...")
        
        print("\n Extracting from all pages...")
        all_csv_files = extract_pdf_tables_to_csv(
            pdf_path=pdf_path,
            output_dir="pdf",
            pages="all",
            flavor="stream"  # Use the flavor that worked
        )
        
        print(f"\n Extraction completed!")
        print(f" Total tables extracted: {len(all_csv_files)}")
        print(f" Files saved in: ./pdf/")
    else:
        print("\n Test failed. Please check the PDF file and try again.")


if __name__ == "__main__":
    main()
