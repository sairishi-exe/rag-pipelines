import json
import math
import time
import requests
import pandas as pd

# https://pmc.ncbi.nlm.nih.gov/tools/id-converter-api/
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
BATCH_SIZE = 200
NCBI_SLEEP = 0.34

from src.utils import cache_data, print_example_row
from src.config import RAW_DATA_PATH, VERBOSE


def load_factoid_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load BioASQ training data and filter to factoid questions only.
    Extracts PMIDs from the document URLs into a separate column.
    Returns a DataFrame with columns: id, body, exact_answer, ideal_answer, pmids
    """
    with open(path) as f:
        data = json.load(f)

    factoids = [q for q in data['questions'] if q['type'] == 'factoid']

    df = pd.DataFrame(factoids)[['id', 'body', 'exact_answer', 'ideal_answer', 'documents']]
    # document URLs are PubMed links, extract the trailing PMID
    df['pmids'] = df['documents'].apply(lambda docs: [d.split('/')[-1] for d in docs])
    df = df.drop(columns=['documents'])

    return df


def pmids_to_pmcids(pmids: list[str], verbose: bool = False) -> dict[str, str]:
    """
    Batch-convert a list of PMIDs to PMCIDs using the NCBI PMC ID Converter API.
    Only PMIDs with an open-access PMC article will have a PMCID returned.
    Returns a dict of {pmid_str: pmcid_str}.
    """
    mapping = {}
    total_batches = math.ceil(len(pmids) / BATCH_SIZE)

    # API accepts up to BATCH_SIZE IDs per request
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        if verbose:
            print(f"Batch {batch_num} / {total_batches}")

        # retry up to 3 times on failed or non-JSON responses
        records = []
        for attempt in range(3):
            response = requests.get(IDCONV_URL, params={"ids": ",".join(batch), "format": "json"})
            try:
                records = response.json().get("records", [])
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            print(f"Skipping batch {batch_num} after 3 failed attempts")

        for record in records:
            pmid = record.get("pmid")
            pmcid = record.get("pmcid")
            # pmid comes back as int from the API, convert to str for consistent keys
            if pmid and pmcid:
                mapping[str(pmid)] = pmcid

        time.sleep(NCBI_SLEEP)

    return mapping


def resolve_pmcids(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Add a pmcids column to the DataFrame by resolving each row's PMIDs via the API.
    Rows where none of the PMIDs have a PMC entry will get an empty list.
    """

    # get all unique PMIDs from the dataframe
    all_pmids = set()
    for pmids in df['pmids']:
        for p in pmids:
            all_pmids.add(p)
    all_pmids = list(all_pmids)

    # convert PMIDs to PMCIDs
    mapping = pmids_to_pmcids(all_pmids, verbose=verbose)

    # add the PMCIDs to the dataframe
    def get_pmcids(pmids: list[str]):
        result = []
        for p in pmids:
            if p in mapping:
                result.append(mapping[p])
        return result

    df['pmcids'] = df['pmids'].apply(get_pmcids)
    return df, mapping


if __name__ == '__main__':

    # step 1: filter and load the data
    df = load_factoid_data(path=RAW_DATA_PATH)

    if VERBOSE:
        print('Factoid count:', len(df), '\n')
        print_example_row(df)

    # step 2: resolve the PMIDs to PMCIDs
    df, mapping = resolve_pmcids(df, verbose=VERBOSE)

    # step 3: cache the data
    cache_data(dataframe_cache=df, document_id_map=mapping)

    if VERBOSE:
        print_example_row(df)

    
