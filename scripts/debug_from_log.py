import re
from collections import Counter
import os

# Path to your extracted log data
file_path = "/data/horse/ws/jixu233b-metadata_ws/hpc_out/2758411.out"

def report_dimension_distribution(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    # Pattern captures the second number inside torch.Size([dim1, dim2, dim3])
    # specifically for freqs_x shape lines
    pattern = r"freqs_x shape:\s+torch\.Size\(\[\d+,\s*(\d+),\s*\d+\]\)"
    
    dim2_values = []

    try:
        with open(path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    # Convert the captured string digits to an integer
                    dim2_values.append(int(match.group(1)))
        
        if not dim2_values:
            print("No matching freqs_x shapes found.")
            return

        # Calculate distribution
        counts = Counter(dim2_values)
        total = len(dim2_values)

        print(f"--- Distribution Report for 2nd Dimension (freqs_x) ---")
        print(f"Total entries processed: {total}")
        print(f"{'Value':<15} | {'Count':<10} | {'Percentage':<10}")
        print("-" * 45)

        # Sort by value (numerical order)
        for val in sorted(counts.keys()):
            count = counts[val]
            percentage = (count / total) * 100
            print(f"{val:<15} | {count:<10} | {percentage:>8.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    report_dimension_distribution(file_path)