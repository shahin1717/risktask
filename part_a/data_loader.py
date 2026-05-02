from datasets import load_dataset, Features, Value

def loader(data_dir: str, split: str = "test"):
    """
    Load Mozilla Common Voice Azerbaijani dataset from local TSV files.
    
    Args:
        data_dir: Path to folder containing train.tsv, dev.tsv, test.tsv and clips/
        split: "train", "validation", or "test"
    
    Returns:
        HuggingFace Dataset
    """

    features = Features({
        "client_id":       Value("string"),
        "path":            Value("string"),
        "sentence_id":     Value("string"),
        "sentence":        Value("string"),
        "sentence_domain": Value("string"),
        "up_votes":        Value("int64"),
        "down_votes":      Value("int64"),
        "age":             Value("string"),
        "gender":          Value("string"),
        "accents":         Value("string"),
        "variant":         Value("string"),
        "locale":          Value("string"),
        "segment":         Value("string"),
    })

    ds = load_dataset(
        "csv",
        data_files={
            "train":      f"{data_dir}/train.tsv",
            "validation": f"{data_dir}/dev.tsv",
            "test":       f"{data_dir}/test.tsv",
        },
        delimiter="\t",
        features=features,
    )

    def add_full_path(example):
        example["path"] = f"{data_dir}/clips/{example['path']}"
        return example

    ds = ds.map(add_full_path)

    return ds[split]