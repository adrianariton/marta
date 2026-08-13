from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="marta_corrected",  # Your local folder name
    repo_id="aariton/MARTA-turn-by-turn",  # HF target repo name
    repo_type="dataset",
    delete_patterns="train-*.parquet",  # Optional: delete existing files matching this pattern
)
