import argparse
import pandas as pd
from datatrust import TrustAnalyzer
from datatrust.explain import explain_column

def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset trustworthiness"
    )
    parser.add_argument(
        "file",
        help="Path to CSV file"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.file)

    analyzer = TrustAnalyzer(df)
    results = analyzer.analyze()

    print("\n🔍 DATA TRUST REPORT\n")

    for col, metrics in results.items():
        print(f"{col}: Trust Score = {metrics['trust_score']}")
        print(" ", explain_column(col, metrics))
        print("-" * 40)

if __name__ == "__main__":
    main()
