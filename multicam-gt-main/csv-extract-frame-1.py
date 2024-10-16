import pandas as pd
import os

def extract_frame_id_1(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Ensure 'frame_id' is of integer type
    df['frame_id'] = df['frame_id'].astype(int)
    
    # Filter the dataframe to keep only rows where frame_id is 1
    df_frame_1 = df[df['frame_id'] == 1]
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the filtered dataframe to a new CSV file
    df_frame_1.to_csv(output_file, index=False)
    
    print(f"CSV file with frame_id 1 has been created: {output_file}")
    print(f"Number of rows: {len(df_frame_1)}")
    
    # Print the first few rows of the extracted data
    print("\nFirst few rows of the extracted data:")
    print(df_frame_1.head())

# Usage
input_file = "../../gtm_hit_annotations.csv"
output_file = "../../gtm_hit_annotations_frame1.csv"

extract_frame_id_1(input_file, output_file)
