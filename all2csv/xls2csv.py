import pandas as pd
import os

def convert_xls_to_csv(xls_path: str, output_dir: str):
    if not os.path.exists(xls_path):
        raise FileNotFoundError(f"File not found: {xls_path}")


    excel_file = pd.ExcelFile(xls_path, engine="xlrd")
    print(f"Found sheets: {excel_file.sheet_names}")

    for sheet in excel_file.sheet_names:
        df = excel_file.parse(sheet)
        

        new_columns = []
        for col in df.columns:
            col_str = str(col)
            if col_str.startswith('Unnamed:') or col_str == 'nan':
                new_columns.append('')
            else:
                new_columns.append(col_str)
        df.columns = new_columns
        
        sheet_name = sheet.replace("/", "_").replace("\\", "_").strip()
        csv_filename = os.path.join(output_dir, f"{sheet_name}.csv")
        df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"Exported sheet '{sheet}' to: {csv_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .xls to .csv by sheet")
    parser.add_argument("xls_path", help="Path to the .xls file")
    parser.add_argument("--output_dir", default=".", help="Directory to save .csv files")
    args = parser.parse_args()

    convert_xls_to_csv(args.xls_path, args.output_dir)
