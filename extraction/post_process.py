import argparse
import ast
import html
import os
import pandas as pd
import re
import unicodedata

from dataclasses import dataclass
from typing import Optional


KEEP_COLUMNS = [
    "user_id", "item_id", "timestamp", "rating", 
    "review_title", "review",
    "statements", "statement_ids", "sentiments"
]


@dataclass
class SurfaceNormConfig:
    subject: str = "the product"
    number_mode: str = "keep"      # {"keep", "mask", "bucket"}
    min_chars: int = 8
    min_tokens: int = 3
    drop_if_only_subject: bool = True


class SurfaceStatementNormalizer:
    _WS = re.compile(r"\s+")
    _URL = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
    _EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)

    # Normalize leading subject variants -> config.subject
    # (applies only at beginning; safer than global replace)
    _LEADING_SUBJECT = re.compile(
        r"^(?:it|this|that|they|these|those|this\s+product|this\s+item|the\s+item|the\s+product)\b\s*",
        re.IGNORECASE
    )

    # Basic punctuation cleanup (keep apostrophes inside words if any)
    _PUNCT_TO_SPACE = re.compile(r"[\/\\\|\(\)\[\]\{\}:;\"“”‘’`~^+=<>]")
    _DASHES = re.compile(r"[\-_–—]+")

    # numbers in words
    _NUM_WORDS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
        "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"
    }
    _NUM_WORD_RE = re.compile(r"\b(" + "|".join(_NUM_WORDS.keys()) + r")\b", re.IGNORECASE)

    _DIGITS = re.compile(r"\b\d+(?:[.,]\d+)?\b")

    # Remove trailing punctuation noise
    _TRAILING_DOTS = re.compile(r"[.!?]+$")

    # Very light stopword-ish trailing filler
    _TRAILING_FILLER = re.compile(r"\b(?:overall|anyway)\b$", re.IGNORECASE)

    def __init__(self, cfg: SurfaceNormConfig):
        self.cfg = cfg

    def _normalize_numbers(self, s: str) -> str:
        # convert small number words -> digits
        def repl(m):
            return self._NUM_WORDS[m.group(1).lower()]
        s = self._NUM_WORD_RE.sub(repl, s)

        if self.cfg.number_mode == "keep":
            return s

        if self.cfg.number_mode == "bucket":
            # bucketize: 1,2,3-5,6-10,10+
            def buck(m):
                x = m.group(0).replace(",", ".")
                try:
                    v = float(x)
                except:
                    return "<num>"
                if v <= 1:
                    return "<n1>"
                if v <= 2:
                    return "<n2>"
                if v <= 5:
                    return "<n3_5>"
                if v <= 10:
                    return "<n6_10>"
                return "<n10p>"
            return self._DIGITS.sub(buck, s)

        # default: mask
        return self._DIGITS.sub("<num>", s)

    def normalize(self, statement: str) -> Optional[str]:
        if statement is None:
            return None

        s = str(statement)

        # HTML/unicode normalization
        s = html.unescape(s)
        s = unicodedata.normalize("NFKC", s)

        # Lowercase early for deterministic matching
        s = s.lower().strip()
        if not s:
            return None

        # Drop URLs / emails
        s = self._URL.sub(" <url> ", s)
        s = self._EMAIL.sub(" <email> ", s)

        # Replace dash-like separators with spaces
        s = self._DASHES.sub(" ", s)

        # Remove some punctuation -> spaces
        s = self._PUNCT_TO_SPACE.sub(" ", s)

        # Normalize commas/periods to spaces (keeps meaning for clustering)
        s = s.replace(",", " ").replace(".", " ")

        # Canonicalize leading subject
        s = self._LEADING_SUBJECT.sub(self.cfg.subject + " ", s).strip()

        # Normalize numbers
        s = self._normalize_numbers(s)

        # Cleanup whitespace + trailing junk
        s = self._WS.sub(" ", s).strip()
        s = self._TRAILING_DOTS.sub("", s).strip()
        s = self._TRAILING_FILLER.sub("", s).strip()
        s = self._WS.sub(" ", s).strip()

        # Cheap validity filters
        if len(s) < self.cfg.min_chars:
            return None
        if len(s.split()) < self.cfg.min_tokens:
            return None
        if self.cfg.drop_if_only_subject and s == self.cfg.subject:
            return None

        return s


def normalize_pair(statement: str, normalizer: SurfaceStatementNormalizer) -> Optional[str]:
    return normalizer.normalize(statement)


def process_dataset(data_df, normalizer: SurfaceStatementNormalizer):
    all_pairs = {}
    cleaned_statements = []
    statement_idss = []
    sentiment_idss = []

    sentiment_map = {"positive": +1, "neutral": 0, "negative": -1}
    
    for index, pair_list in enumerate(data_df["cleaned"].tolist(), start=1):
        try:
            pair_list = ast.literal_eval(pair_list)
            if isinstance(pair_list, dict):
                pair_list = pair_list.get("statements", [])
            else:
                pair_list = list(pair_list)
        except:
            cleaned_statements.append(None)
            statement_idss.append(None)
            sentiment_idss.append(None)
            continue

        new_pair_list = []
        statement_ids = []
        sentiment_ids = []
        
        for pair in pair_list:
            if not pair or not isinstance(pair, dict):
                print("Invalid pair:", pair)
                continue

            sentiment = pair.get("sentiment")
            if sentiment: sentiment = sentiment.lower().strip()
            if sentiment not in sentiment_map: continue
                
            statement = pair.get("statement")
            if statement: statement = normalizer.normalize(statement)
            if not statement: continue

            pair_tuple = (statement, sentiment)
            if pair_tuple not in all_pairs:
                new_id = len(all_pairs)
                all_pairs[pair_tuple] = {}
                all_pairs[pair_tuple]["id"] = new_id
                all_pairs[pair_tuple]["frequency"] = 0
                
            all_pairs[pair_tuple]["frequency"] += 1

            sentiment_id = sentiment_map[sentiment]
            statement_id = all_pairs[pair_tuple]["id"]

            new_pair = {"statement": statement, "sentiment": sentiment}
            new_pair_list.append(new_pair)
            statement_ids.append(statement_id)
            sentiment_ids.append(sentiment_id)

        if len(new_pair_list) == 0:
            new_pair_list = None
            statement_ids = None
            sentiment_ids = None
            
        cleaned_statements.append(new_pair_list)
        statement_idss.append(statement_ids)
        sentiment_idss.append(sentiment_ids)

        if index % 10_000 == 0:
            print("10000 samples processed...")

    print("Done!")
    n_none = cleaned_statements.count(None)
    print("Number of fails:", n_none)
    print("% of fails:", n_none/len(data_df))

    new_data_df = pd.DataFrame(data_df)
    new_data_df["statements"] = cleaned_statements
    new_data_df["statement_ids"] = statement_idss
    new_data_df["sentiments"] = sentiment_idss
        
    columns = ["user_id", "item_id", "timestamp", "rating", "statements", "statement_ids", "sentiments"]
    new_data_df = new_data_df.dropna(subset=columns)

    new_data_df = new_data_df[KEEP_COLUMNS]

    return new_data_df, all_pairs


def main(args):
    dataset_dir = args.dataset_dir
    data_df = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"))

    print(f"Dataset Shape: {data_df.shape}")
    print(data_df.head())

    statements = data_df["cleaned"].tolist()
    print("Sample Statements:")
    print(*statements[0:5], sep="\n")

    norm_cfg = SurfaceNormConfig(
        subject="the product",
        number_mode="keep",
        min_chars=8,
        min_tokens=3,
        drop_if_only_subject=True
    )
    normalizer = SurfaceStatementNormalizer(norm_cfg)

    new_data_df, all_pairs = process_dataset(data_df, normalizer)
    print("Processed DataFrame:")
    print(new_data_df.head())

    all_statements = []
    all_sentiments = []
    all_freq = []

    for i, ((statement, sent), id_freq) in enumerate(all_pairs.items()):
        freq = id_freq["frequency"]
        all_statements.append(statement)
        all_sentiments.append(sent)
        all_freq.append(freq)
        if i % 10_000 == 0:
            print((statement, sent), freq)

    statements_df = pd.DataFrame({
        "statement": all_statements,
        "sentiment": all_sentiments,
        "frequency": all_freq
    })

    print("Statements DataFrame:")
    print(len(statements_df))
    print(statements_df.sample(n=10))

    new_data_df.to_csv(os.path.join(dataset_dir, "dataset_vS.csv"))
    statements_df.to_csv(os.path.join(dataset_dir, "statements_vS.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()
    main(args)
    