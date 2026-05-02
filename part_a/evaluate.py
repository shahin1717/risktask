import pandas as pd
from jiwer import wer, cer

def compute_metrics(results_csv: str = "results/inference_results.csv"):
    """
    Compute WER and CER from inference results.
    Prints: avg WER, avg CER, best 5 and worst 5 samples.
    """

    df = pd.read_csv(results_csv)

    # Drop any rows with missing values
    df = df.dropna(subset=["reference", "hypothesis"])

    # Compute per-sample WER and CER
    df["wer"] = df.apply(lambda r: wer(r["reference"], r["hypothesis"]), axis=1)
    df["cer"] = df.apply(lambda r: cer(r["reference"], r["hypothesis"]), axis=1)

    # Average metrics
    avg_wer = df["wer"].mean() * 100
    avg_cer = df["cer"].mean() * 100

    print("=" * 60)
    print(f"  Average WER : {avg_wer:.2f}%")
    print(f"  Average CER : {avg_cer:.2f}%")
    print("=" * 60)

    # Best 5 (lowest WER)
    best5 = df.nsmallest(5, "wer")[["reference", "hypothesis", "wer", "cer"]]
    print("\n✅ Best 5 Samples (lowest WER):")
    print("-" * 60)
    for i, row in best5.iterrows():
        print(f"  REF : {row['reference']}")
        print(f"  HYP : {row['hypothesis']}")
        print(f"  WER : {row['wer']*100:.2f}%   CER : {row['cer']*100:.2f}%")
        print()

    # Worst 5 (highest WER)
    worst5 = df.nlargest(5, "wer")[["reference", "hypothesis", "wer", "cer"]]
    print("❌ Worst 5 Samples (highest WER):")
    print("-" * 60)
    for i, row in worst5.iterrows():
        print(f"  REF : {row['reference']}")
        print(f"  HYP : {row['hypothesis']}")
        print(f"  WER : {row['wer']*100:.2f}%   CER : {row['cer']*100:.2f}%")
        print()

    # Save full results with metrics
    df.to_csv("results/evaluated_results.csv", index=False)
    print("Full results saved to results/evaluated_results.csv")

    return avg_wer, avg_cer


if __name__ == "__main__":
    compute_metrics()