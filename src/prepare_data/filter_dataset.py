import pandas as pd

from src.config import OA_PDF_CSV, TARGET_ARTICLES, VERBOSE
from src.utils import load_from_cache, cache_data


def load_pdf_pmcids(csv_path: str) -> set:
    """
    Load the oa_non_comm_use_pdf.csv and return a set of PMCIDs
    that have a directly downloadable PDF.
    """
    df = pd.read_csv(csv_path, usecols=["Accession ID"])
    return set(df["Accession ID"].str.strip())


def compute_qa_stats(df: pd.DataFrame, pdf_pmcids: set) -> pd.DataFrame:
    """
    Adds two columns:
      pmcid_oa: subset of pmcids that have a downloadable PDF
      oa_ratio: len(pmcid_oa) / len(pmids)
      
    Returns df sorted by oa_ratio descending.
    """
    df["pmcid_oa"] = df["pmcids"].apply(
        lambda pmcids: [p for p in pmcids if p in pdf_pmcids]
    )
    df["oa_ratio"] = df.apply(
        lambda row: len(row["pmcid_oa"]) / len(row["pmids"]) if row["pmids"] else 0,
        axis=1,
    )
    return df.sort_values("oa_ratio", ascending=False).reset_index(drop=True)


def select_subset(
    df: pd.DataFrame, target_articles: int = TARGET_ARTICLES, min_oa_ratio: float = 0.5
) -> tuple[pd.DataFrame, set]:
    """Select a subset of QA pairs and their associated documents.
    First pass collects unique PMCIDs until we hit the target count.
    Second pass includes any remaining rows already covered by those documents.
    Returns (filtered_df, unique_pmcids)."""
    candidates = df[df["oa_ratio"] >= min_oa_ratio]

    # pass 1: build the corpus from downloadable PDFs only
    unique_pmcids = set()
    for _, row in candidates.iterrows():
        new_pmcids = [p for p in row["pmcid_oa"] if p not in unique_pmcids]
        if not new_pmcids:
            continue
        unique_pmcids.update(new_pmcids)
        if len(unique_pmcids) >= target_articles:
            break

    # pass 2: include all rows whose pmcid_oa are fully covered by the corpus
    # row["pmcid_oa"] must be non-empty (empty list would pass the all() check)
    selected_rows = [
        row for _, row in candidates.iterrows()
        if row["pmcid_oa"] and all(p in unique_pmcids for p in row["pmcid_oa"])
    ]

    return pd.DataFrame(selected_rows).reset_index(drop=True), unique_pmcids


if __name__ == "__main__":
    # load full dataset from previous build step
    df, document_id_map = load_from_cache("dataframe_cache", "document_id_map")

    # load list of PMCIDs that have downloadable PDFs
    pdf_pmcids = load_pdf_pmcids(OA_PDF_CSV)

    # add OA ratio columns to each question
    df = compute_qa_stats(df, pdf_pmcids)

    # select subset of questions and documents
    df, unique_pmcids = select_subset(df, target_articles=TARGET_ARTICLES)

    # save filtered dataset to cache
    cache_data(
        dataframe_subset_cache=df,
        document_id_subset={"pmcids": list(unique_pmcids)},
    )

    if VERBOSE:
        print(f"QA pairs:              {len(df)}")
        print(f"Unique OA documents:   {len(unique_pmcids)}")
        print(f"Mean OA docs per pair: {df['pmcid_oa'].apply(len).mean():.1f}")
