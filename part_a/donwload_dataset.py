from datasets import load_dataset, Audio,Features,Value

DATA_DIR = "data"  # change to your folder

features = Features({
    "client_id": Value("string"),
    "path": Value("string"),
    "sentence_id": Value("string"),
    "sentence": Value("string"),
    "sentence_domain": Value("string"),
    "up_votes": Value("int64"),
    "down_votes": Value("int64"),
    "age": Value("string"),
    "gender": Value("string"),
    "accents": Value("string"),
    "variant": Value("string"),
    "locale": Value("string"),
    "segment": Value("string"),
})


ds = load_dataset(
    "csv",
    data_files={
        "train": f"{DATA_DIR}/train.tsv",
        "validation": f"{DATA_DIR}/dev.tsv",
        "test": f"{DATA_DIR}/test.tsv"
    },
    delimiter="\t",
    features=features
)

# Fix path to actual audio files
def add_full_path(example):
    example["path"] = f"{DATA_DIR}/clips/{example['path']}"
    return example

ds = ds.map(add_full_path)

# Add audio column
# ds = ds.cast_column("path", Audio(sampling_rate=16000))

print(ds["train"][0]["path"])
