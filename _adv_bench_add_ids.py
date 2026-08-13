# import pandas as pd

# # Define the file path
# file_path = "local/walledai/AdvBench.csv"

# # Load the data
# df = pd.read_csv(file_path)

# # Create the 'id' column with the format adv_001, adv_002, etc.
# # we use zfill(3) to ensure the numbers are padded with zeros
# df.insert(0, "id", ["adv_{}".format(str(i + 1).zfill(3)) for i in range(len(df))])

# # Overwrite the original file
# df.to_csv(file_path, index=False)

# print(f"Successfully updated {file_path} with {len(df)} IDs.")
