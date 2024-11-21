import pandas as pd
import os

def sort_csv_by_frame_id(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Ensure 'frame_id' is of integer type
    df['frame_id'] = df['frame_id'].astype(int)
    
    # Sort the dataframe by frame_id
    df_sorted = df.sort_values('frame_id')
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the sorted dataframe to a new CSV file
    df_sorted.to_csv(output_file, index=False)
    
    print(f"Sorted CSV file has been created: {output_file}")
    print(f"Number of rows: {len(df_sorted)}")
    print(f"Frame ID range: {df_sorted['frame_id'].min()} to {df_sorted['frame_id'].max()}")

# Usage
input_file = "../../gtm_hit_annotations_frame1.csv"
output_file = "../../gtm_hit_annotations_sorted.csv"

sort_csv_by_frame_id(input_file, output_file)
