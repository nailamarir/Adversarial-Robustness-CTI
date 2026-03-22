"""
Data Preprocessing Module
Handles loading, cleaning, and preparing the AnnoCTR dataset
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocessor for CTI dataset with label reduction and balancing"""

    def __init__(
        self,
        base_path: str,
        top_k_labels: int = 15,
        other_sample_size: int = 600,
        random_state: int = 42
    ):
        self.base_path = base_path
        self.top_k_labels = top_k_labels
        self.other_sample_size = other_sample_size
        self.random_state = random_state

        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.freq_labels: List = []

    @staticmethod
    def _load_jsonl(path: str) -> pd.DataFrame:
        """Load a JSONL file, skipping blank lines."""
        import json
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def load_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load label files from JSONL format.

        Uses *_w_con variants (with extra surrounding context) when available
        to augment training data, roughly doubling effective training size.
        """
        labels_dir = f"{self.base_path}/linking_mitre_only"

        base_train = self._load_jsonl(f"{labels_dir}/train.jsonl")
        # Augment train with context-enriched variant if present
        con_path = f"{labels_dir}/train_w_con.jsonl"
        if os.path.exists(con_path):
            con_train = self._load_jsonl(con_path)
            labels_train = pd.concat([base_train, con_train], ignore_index=True)
            print(f"Augmented train with _w_con: {len(base_train)} + {len(con_train)} = {len(labels_train)}")
        else:
            labels_train = base_train

        # Add hard negatives: label_id=1906 is "No Annotation" (true negatives),
        # force those to OTHER; keep real MITRE IDs as-is (extra training signal)
        neg_path = f"{labels_dir}/train_w_con_w_neg.jsonl"
        if os.path.exists(neg_path):
            neg_train = self._load_jsonl(neg_path)
            neg_train["label_id"] = neg_train["label_id"].astype(object)
            neg_train.loc[neg_train["label_id"] == 1906, "label_id"] = "OTHER"
            labels_train = pd.concat([labels_train, neg_train], ignore_index=True)
            print(f"Added hard negatives: +{len(neg_train)} samples -> {len(labels_train)} total")

        # Step 1: use dev_w_con for evaluation (aligns with rich-context training format)
        dev_path = f"{labels_dir}/dev_w_con.jsonl"
        labels_dev = self._load_jsonl(dev_path if os.path.exists(dev_path) else f"{labels_dir}/dev.jsonl")
        labels_test  = self._load_jsonl(f"{labels_dir}/test.jsonl")

        print(f"Loaded labels - Train: {labels_train.shape}, Dev: {labels_dev.shape}, Test: {labels_test.shape}")
        return labels_train, labels_dev, labels_test

    def load_texts(self) -> pd.DataFrame:
        """Load raw CTI text files"""
        text_dir = f"{self.base_path}/all/text"
        text_rows = []

        for fname in os.listdir(text_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(text_dir, fname), "r", encoding="utf-8") as f:
                    content = f.read()
                text_rows.append({
                    "document": fname,
                    "full_text": content
                })

        text_df = pd.DataFrame(text_rows)
        print(f"Loaded {len(text_df)} text documents")
        return text_df

    @staticmethod
    def clean_name(s: str) -> str:
        """Normalize document names for matching"""
        return s.replace(".txt", "").strip().lower()

    def align_and_merge(
        self,
        labels_df: pd.DataFrame,
        text_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Align labels with text documents"""
        # Clean names
        text_df = text_df.copy()
        labels_df = labels_df.copy()

        text_df["doc_clean"] = text_df["document"].apply(self.clean_name)
        labels_df["doc_clean"] = labels_df["document"].apply(self.clean_name)

        # Filter to available texts
        available = set(text_df["doc_clean"])
        labels_filtered = labels_df[labels_df["doc_clean"].isin(available)]

        # Merge
        merged = labels_filtered.merge(
            text_df[["doc_clean", "full_text"]],
            on="doc_clean",
            how="left"
        )

        return merged

    def reduce_label_space(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Reduce labels to top-K + OTHER"""
        df = df.copy()

        if is_train:
            # Compute frequency from training data
            label_counts = df["label_id"].value_counts()
            self.freq_labels = label_counts.nlargest(self.top_k_labels).index.tolist()
            print(f"Top {self.top_k_labels} labels: {self.freq_labels}")

        # Map rare labels to OTHER
        df["label_mapped"] = df["label_id"].apply(
            lambda x: str(x) if x in self.freq_labels else "OTHER"
        )

        return df

    def build_label_encodings(self, *dfs: pd.DataFrame) -> None:
        """Build label to ID mappings from all datasets"""
        all_labels = pd.concat([df["label_mapped"] for df in dfs]).unique()

        self.label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        print(f"Total unique labels: {len(self.label2id)}")
        print(f"Label mapping: {self.label2id}")

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode string labels to integers"""
        df = df.copy()
        df["label_id_encoded"] = df["label_mapped"].map(self.label2id)

        # Drop any NaN and convert to int
        df = df.dropna(subset=["label_id_encoded"]).reset_index(drop=True)
        df["label_id_encoded"] = df["label_id_encoded"].astype(int)

        return df

    def balance_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the OTHER class in training data"""
        other_samples = df[df["label_mapped"] == "OTHER"]
        non_other_samples = df[df["label_mapped"] != "OTHER"]

        # Downsample OTHER
        n_keep = min(self.other_sample_size, len(other_samples))
        other_downsampled = other_samples.sample(n=n_keep, random_state=self.random_state)

        # Combine and shuffle
        balanced = pd.concat([non_other_samples, other_downsampled])
        balanced = balanced.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        print(f"Balanced training data: {len(balanced)} samples")
        print(f"OTHER class: {n_keep} samples ({n_keep / len(balanced) * 100:.1f}%)")

        return balanced

    @staticmethod
    def build_rich_text(df: pd.DataFrame) -> pd.Series:
        """Build rich input text: full sentence context around the mention.

        Uses sentence_left + mention + sentence_right when available,
        falling back to context_left + mention + context_right.
        This gives ~50-100 words of context vs ~17 words previously.
        """
        def _combine(row):
            mention = str(row.get("mention", "") or "")
            # Prefer full sentence context (more signal)
            left  = str(row.get("sentence_left",  "") or "").strip()
            right = str(row.get("sentence_right", "") or "").strip()
            if not left and not right:
                # Fallback to shorter context window
                left  = str(row.get("context_left",  "") or "").strip()
                right = str(row.get("context_right", "") or "").strip()
            parts = []
            if left:
                parts.append(left)
            parts.append(mention)
            if right:
                parts.append(right)
            return " ".join(parts).strip()

        return df.apply(_combine, axis=1)

    def keep_core_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only necessary columns"""
        core_cols = [
            "document", "full_text", "label", "label_title",
            "entity_type", "label_id", "label_mapped", "label_id_encoded"
        ]
        available_cols = [c for c in core_cols if c in df.columns]
        return df[available_cols]

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Full preprocessing pipeline"""
        print("=" * 60)
        print("Starting Data Preprocessing Pipeline")
        print("=" * 60)

        # Load data
        labels_train, labels_dev, labels_test = self.load_labels()
        text_df = self.load_texts()

        # Build rich input text from sentence context (no document merge needed)
        print("\nBuilding rich sentence-context inputs...")
        for df_ref in (labels_train, labels_dev, labels_test):
            df_ref["full_text"] = self.build_rich_text(df_ref)
        train_df, dev_df, test_df = labels_train, labels_dev, labels_test

        print(f"Merged - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
        avg_len = train_df["full_text"].str.split().str.len().mean()
        print(f"Avg input length: {avg_len:.0f} words")

        # Reduce label space
        print("\nReducing label space...")
        train_df = self.reduce_label_space(train_df, is_train=True)
        dev_df = self.reduce_label_space(dev_df, is_train=False)
        test_df = self.reduce_label_space(test_df, is_train=False)

        # Build encodings
        print("\nBuilding label encodings...")
        self.build_label_encodings(train_df, dev_df, test_df)

        # Encode labels
        train_df = self.encode_labels(train_df)
        dev_df = self.encode_labels(dev_df)
        test_df = self.encode_labels(test_df)

        # Balance training data
        print("\nBalancing training data...")
        train_df = self.balance_training_data(train_df)

        # Keep core columns
        train_df = self.keep_core_columns(train_df)
        dev_df = self.keep_core_columns(dev_df)
        test_df = self.keep_core_columns(test_df)

        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print(f"Final sizes - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
        print("=" * 60)

        return train_df, dev_df, test_df

    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Return label mappings"""
        return self.label2id, self.id2label

    def get_statistics(self, df: pd.DataFrame, name: str = "Dataset") -> Dict:
        """Get dataset statistics"""
        stats = {
            "name": name,
            "total_samples": len(df),
            "unique_labels": df["label_mapped"].nunique(),
            "label_distribution": df["label_mapped"].value_counts().to_dict(),
            "avg_text_length": df["full_text"].str.len().mean(),
            "max_text_length": df["full_text"].str.len().max(),
            "min_text_length": df["full_text"].str.len().min(),
        }
        return stats
