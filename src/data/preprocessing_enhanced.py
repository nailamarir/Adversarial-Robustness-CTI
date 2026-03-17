"""
Enhanced Data Preprocessing Module
Improved preprocessing with text cleaning, augmentation, and better balancing
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from collections import Counter
import random


class EnhancedDataPreprocessor:
    """
    Enhanced Preprocessor for CTI dataset with:
    - Text cleaning and normalization
    - Smart truncation strategies
    - Better class balancing (oversampling + undersampling)
    - Data augmentation
    - Stratified splitting
    """

    def __init__(
        self,
        base_path: str,
        top_k_labels: int = 15,
        min_samples_per_class: int = 50,
        max_samples_per_class: int = 800,
        random_state: int = 42,
        clean_text: bool = True,
        augment_data: bool = True,
        augmentation_factor: float = 0.3
    ):
        self.base_path = base_path
        self.top_k_labels = top_k_labels
        self.min_samples_per_class = min_samples_per_class
        self.max_samples_per_class = max_samples_per_class
        self.random_state = random_state
        self.clean_text = clean_text
        self.augment_data = augment_data
        self.augmentation_factor = augmentation_factor

        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.freq_labels: List = []

        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)

    # ==================== TEXT CLEANING ====================

    def clean_cti_text(self, text: str) -> str:
        """
        Clean CTI text while preserving important technical content

        Improvements:
        - Remove URLs but keep domain names as they can be indicators
        - Normalize whitespace
        - Remove excessive special characters
        - Preserve technical terms and IOCs
        """
        if not isinstance(text, str):
            return ""

        # Store original for fallback
        original_length = len(text)

        # 1. Extract and preserve domain names from URLs
        url_pattern = r'https?://([a-zA-Z0-9.-]+)(?:/[^\s]*)?'
        domains = re.findall(url_pattern, text)

        # 2. Remove full URLs but we'll add domains back
        text = re.sub(r'https?://[^\s]+', ' [URL] ', text)

        # 3. Remove email addresses but mark them
        text = re.sub(r'\S+@\S+\.\S+', ' [EMAIL] ', text)

        # 4. Normalize file paths (keep filename)
        text = re.sub(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*([^\\/:*?"<>|\r\n]+)', r' \1 ', text)
        text = re.sub(r'/(?:[^/\s]+/)*([^/\s]+\.[a-zA-Z]+)', r' \1 ', text)

        # 5. Preserve IP addresses (important IOCs)
        # Already preserved, just normalize spacing

        # 6. Remove code blocks but keep a marker
        text = re.sub(r'```[\s\S]*?```', ' [CODE_BLOCK] ', text)
        text = re.sub(r'`[^`]+`', ' [CODE] ', text)

        # 7. Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\'\"\/\@\#\$\%\&\*\+\=\<\>\?\!]', ' ', text)

        # 8. Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # 9. Remove very short lines that are likely noise
        lines = text.split('\n')
        lines = [l.strip() for l in lines if len(l.strip()) > 10]
        text = ' '.join(lines)

        # 10. Lowercase for consistency (BERT/SecBERT handle this)
        # Note: Keep case for proper nouns in CTI
        # text = text.lower()  # Commented: models usually handle this

        # 11. Add back unique domains as context
        if domains:
            unique_domains = list(set(domains))[:5]  # Keep max 5
            text = text + " [DOMAINS: " + ", ".join(unique_domains) + "]"

        # Safety check: don't return empty text
        if len(text.strip()) < 50 and original_length > 50:
            return text[:original_length]  # Return less aggressively cleaned

        return text.strip()

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent processing

        - Standardize quotes
        - Normalize unicode characters
        - Handle common CTI abbreviations
        """
        if not isinstance(text, str):
            return ""

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')

        # Common CTI abbreviations expansion (optional, helps with understanding)
        abbreviations = {
            'C2': 'command and control',
            'C&C': 'command and control',
            'APT': 'advanced persistent threat',
            'IOC': 'indicator of compromise',
            'IOCs': 'indicators of compromise',
            'TTPs': 'tactics techniques and procedures',
            'RAT': 'remote access trojan',
            'DDoS': 'distributed denial of service',
        }

        for abbr, expansion in abbreviations.items():
            # Keep both abbreviation and expansion for context
            text = re.sub(
                rf'\b{abbr}\b',
                f'{abbr} ({expansion})',
                text,
                flags=re.IGNORECASE
            )

        return text

    # ==================== SMART TRUNCATION ====================

    def smart_truncate(
        self,
        text: str,
        max_chars: int = 4000,  # ~512 tokens * 8 chars avg
        strategy: str = "head_tail"
    ) -> str:
        """
        Smart truncation strategies for long CTI documents

        Strategies:
        - "head": Keep first N characters (original behavior)
        - "tail": Keep last N characters
        - "head_tail": Keep first half + last half (captures intro and conclusions)
        - "important": Extract sentences with key CTI terms
        """
        if len(text) <= max_chars:
            return text

        if strategy == "head":
            return text[:max_chars]

        elif strategy == "tail":
            return text[-max_chars:]

        elif strategy == "head_tail":
            half = max_chars // 2
            return text[:half] + " [...] " + text[-half:]

        elif strategy == "important":
            # Extract sentences containing important CTI keywords
            important_keywords = [
                'attack', 'malware', 'vulnerability', 'exploit', 'threat',
                'compromise', 'breach', 'payload', 'backdoor', 'trojan',
                'ransomware', 'phishing', 'spear', 'lateral', 'persistence',
                'exfiltration', 'command', 'control', 'execution', 'discovery'
            ]

            sentences = re.split(r'(?<=[.!?])\s+', text)
            scored_sentences = []

            for sent in sentences:
                score = sum(1 for kw in important_keywords if kw.lower() in sent.lower())
                scored_sentences.append((score, sent))

            # Sort by score (descending) and take top sentences
            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            result = []
            current_length = 0
            for score, sent in scored_sentences:
                if current_length + len(sent) < max_chars:
                    result.append(sent)
                    current_length += len(sent)
                else:
                    break

            # Return in original order if possible
            return ' '.join(result) if result else text[:max_chars]

        return text[:max_chars]

    # ==================== DATA AUGMENTATION ====================

    def augment_text(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Generate augmented versions of CTI text

        Techniques:
        - Synonym replacement (CTI-aware)
        - Random deletion of non-essential words
        - Sentence shuffling (for multi-sentence texts)
        """
        augmented = []

        for _ in range(num_augmentations):
            aug_type = random.choice(['synonym', 'deletion', 'shuffle'])

            if aug_type == 'synonym':
                augmented.append(self._synonym_replacement(text))
            elif aug_type == 'deletion':
                augmented.append(self._random_deletion(text))
            elif aug_type == 'shuffle':
                augmented.append(self._sentence_shuffle(text))

        return augmented

    def _synonym_replacement(self, text: str, n_replacements: int = 3) -> str:
        """Replace words with CTI-relevant synonyms"""
        # CTI-specific synonym dictionary
        cti_synonyms = {
            'attack': ['assault', 'offensive', 'strike', 'intrusion'],
            'malware': ['malicious software', 'virus', 'threat'],
            'hacker': ['threat actor', 'adversary', 'attacker'],
            'steal': ['exfiltrate', 'extract', 'harvest'],
            'install': ['deploy', 'drop', 'plant'],
            'execute': ['run', 'launch', 'invoke'],
            'target': ['victim', 'objective', 'focus'],
            'spread': ['propagate', 'distribute', 'disseminate'],
            'hide': ['conceal', 'obfuscate', 'mask'],
            'detect': ['identify', 'discover', 'find'],
        }

        words = text.split()
        for _ in range(n_replacements):
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?;:')
                if word_lower in cti_synonyms:
                    words[i] = random.choice(cti_synonyms[word_lower])
                    break

        return ' '.join(words)

    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete non-essential words"""
        # Words to preserve (CTI-important)
        preserve = {
            'malware', 'attack', 'threat', 'vulnerability', 'exploit',
            'backdoor', 'trojan', 'ransomware', 'phishing', 'apt',
            'command', 'control', 'execute', 'payload', 'persistence'
        }

        words = text.split()
        if len(words) <= 10:
            return text

        new_words = []
        for word in words:
            if word.lower() in preserve or random.random() > p:
                new_words.append(word)

        return ' '.join(new_words) if new_words else text

    def _sentence_shuffle(self, text: str) -> str:
        """Shuffle sentences while keeping first and last"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 3:
            return text

        first = sentences[0]
        last = sentences[-1]
        middle = sentences[1:-1]
        random.shuffle(middle)

        return ' '.join([first] + middle + [last])

    # ==================== CLASS BALANCING ====================

    def balance_classes(
        self,
        df: pd.DataFrame,
        label_column: str = "label_mapped"
    ) -> pd.DataFrame:
        """
        Advanced class balancing using combination of:
        - Undersampling majority classes
        - Oversampling minority classes
        - SMOTE-like augmentation for text

        This addresses the issue of having some classes with very few samples
        while OTHER class has too many.
        """
        class_counts = df[label_column].value_counts()
        print(f"\nOriginal class distribution:")
        print(class_counts)

        balanced_dfs = []

        for label in class_counts.index:
            label_df = df[df[label_column] == label].copy()
            current_count = len(label_df)

            if current_count > self.max_samples_per_class:
                # Undersample: randomly select max_samples
                label_df = label_df.sample(
                    n=self.max_samples_per_class,
                    random_state=self.random_state
                )
                print(f"  {label}: {current_count} → {self.max_samples_per_class} (undersampled)")

            elif current_count < self.min_samples_per_class and self.augment_data:
                # Oversample with augmentation
                samples_needed = self.min_samples_per_class - current_count
                augmented_rows = []

                for _ in range(samples_needed):
                    # Randomly select a sample to augment
                    sample = label_df.sample(n=1, random_state=None).iloc[0]
                    new_row = sample.copy()

                    # Augment the text
                    aug_texts = self.augment_text(sample['full_text'], num_augmentations=1)
                    new_row['full_text'] = aug_texts[0]
                    new_row['is_augmented'] = True

                    augmented_rows.append(new_row)

                if augmented_rows:
                    aug_df = pd.DataFrame(augmented_rows)
                    label_df = pd.concat([label_df, aug_df], ignore_index=True)
                    print(f"  {label}: {current_count} → {len(label_df)} (augmented)")
            else:
                print(f"  {label}: {current_count} (unchanged)")

            balanced_dfs.append(label_df)

        balanced = pd.concat(balanced_dfs, ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        print(f"\nBalanced total: {len(balanced)} samples")
        return balanced

    # ==================== CORE PREPROCESSING ====================

    def load_labels(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load label files from JSONL format"""
        labels_dir = f"{self.base_path}/linking_mitre_only"

        labels_train = pd.read_json(f"{labels_dir}/train.jsonl", lines=True)
        labels_dev = pd.read_json(f"{labels_dir}/dev.jsonl", lines=True)
        labels_test = pd.read_json(f"{labels_dir}/test.jsonl", lines=True)

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
        text_df = text_df.copy()
        labels_df = labels_df.copy()

        text_df["doc_clean"] = text_df["document"].apply(self.clean_name)
        labels_df["doc_clean"] = labels_df["document"].apply(self.clean_name)

        available = set(text_df["doc_clean"])
        labels_filtered = labels_df[labels_df["doc_clean"].isin(available)]

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
            label_counts = df["label_id"].value_counts()
            self.freq_labels = label_counts.nlargest(self.top_k_labels).index.tolist()
            print(f"Top {self.top_k_labels} labels: {self.freq_labels}")

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

        df = df.dropna(subset=["label_id_encoded"]).reset_index(drop=True)
        df["label_id_encoded"] = df["label_id_encoded"].astype(int)

        return df

    def preprocess_texts(
        self,
        df: pd.DataFrame,
        truncation_strategy: str = "head_tail"
    ) -> pd.DataFrame:
        """Apply text cleaning and truncation"""
        df = df.copy()

        if self.clean_text:
            print("Cleaning texts...")
            df["full_text"] = df["full_text"].apply(self.clean_cti_text)
            df["full_text"] = df["full_text"].apply(self.normalize_text)

        # Apply smart truncation
        print(f"Applying {truncation_strategy} truncation...")
        df["full_text"] = df["full_text"].apply(
            lambda x: self.smart_truncate(x, strategy=truncation_strategy)
        )

        return df

    def keep_core_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only necessary columns"""
        core_cols = [
            "document", "full_text", "label", "label_title",
            "entity_type", "label_id", "label_mapped", "label_id_encoded",
            "is_augmented"
        ]
        available_cols = [c for c in core_cols if c in df.columns]
        return df[available_cols]

    def process(
        self,
        truncation_strategy: str = "head_tail"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full enhanced preprocessing pipeline

        Args:
            truncation_strategy: One of "head", "tail", "head_tail", "important"

        Returns:
            Processed train, dev, test DataFrames
        """
        print("=" * 60)
        print("Starting ENHANCED Data Preprocessing Pipeline")
        print("=" * 60)
        print(f"Settings:")
        print(f"  - Text cleaning: {self.clean_text}")
        print(f"  - Data augmentation: {self.augment_data}")
        print(f"  - Truncation strategy: {truncation_strategy}")
        print(f"  - Min samples/class: {self.min_samples_per_class}")
        print(f"  - Max samples/class: {self.max_samples_per_class}")
        print("=" * 60)

        # Load data
        labels_train, labels_dev, labels_test = self.load_labels()
        text_df = self.load_texts()

        # Align and merge
        print("\nAligning labels with text documents...")
        train_df = self.align_and_merge(labels_train, text_df)
        dev_df = self.align_and_merge(labels_dev, text_df)
        test_df = self.align_and_merge(labels_test, text_df)

        print(f"Merged - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

        # Clean and preprocess texts
        print("\nPreprocessing texts...")
        train_df = self.preprocess_texts(train_df, truncation_strategy)
        dev_df = self.preprocess_texts(dev_df, truncation_strategy)
        test_df = self.preprocess_texts(test_df, truncation_strategy)

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

        # Balance training data (enhanced method)
        print("\nBalancing training data (enhanced)...")
        train_df['is_augmented'] = False
        train_df = self.balance_classes(train_df)

        # Keep core columns
        train_df = self.keep_core_columns(train_df)
        dev_df = self.keep_core_columns(dev_df)
        test_df = self.keep_core_columns(test_df)

        print("\n" + "=" * 60)
        print("Enhanced Preprocessing Complete!")
        print(f"Final sizes - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

        # Print final class distribution
        print("\nFinal training class distribution:")
        print(train_df["label_mapped"].value_counts())
        print("=" * 60)

        return train_df, dev_df, test_df

    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Return label mappings"""
        return self.label2id, self.id2label

    def get_statistics(self, df: pd.DataFrame, name: str = "Dataset") -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            "name": name,
            "total_samples": len(df),
            "unique_labels": df["label_mapped"].nunique(),
            "label_distribution": df["label_mapped"].value_counts().to_dict(),
            "avg_text_length": df["full_text"].str.len().mean(),
            "max_text_length": df["full_text"].str.len().max(),
            "min_text_length": df["full_text"].str.len().min(),
            "std_text_length": df["full_text"].str.len().std(),
        }

        if "is_augmented" in df.columns:
            stats["augmented_samples"] = df["is_augmented"].sum()
            stats["original_samples"] = len(df) - df["is_augmented"].sum()

        return stats


# Comparison function
def compare_preprocessing_methods(base_path: str):
    """
    Compare original vs enhanced preprocessing

    Returns statistics for both methods
    """
    from .preprocessing import DataPreprocessor

    print("=" * 70)
    print("PREPROCESSING COMPARISON")
    print("=" * 70)

    # Original preprocessing
    print("\n[1/2] Running ORIGINAL preprocessing...")
    original_prep = DataPreprocessor(
        base_path=base_path,
        top_k_labels=15,
        other_sample_size=600
    )
    orig_train, orig_dev, orig_test = original_prep.process()

    # Enhanced preprocessing
    print("\n[2/2] Running ENHANCED preprocessing...")
    enhanced_prep = EnhancedDataPreprocessor(
        base_path=base_path,
        top_k_labels=15,
        min_samples_per_class=50,
        max_samples_per_class=800,
        clean_text=True,
        augment_data=True
    )
    enh_train, enh_dev, enh_test = enhanced_prep.process()

    # Compare statistics
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n📊 Training Set Comparison:")
    print(f"  Original:  {len(orig_train)} samples")
    print(f"  Enhanced:  {len(enh_train)} samples")

    print("\n📊 Class Balance (training):")
    orig_dist = orig_train["label_mapped"].value_counts()
    enh_dist = enh_train["label_mapped"].value_counts()

    print(f"  Original std dev:  {orig_dist.std():.2f}")
    print(f"  Enhanced std dev:  {enh_dist.std():.2f}")

    print("\n📊 Text Length (training):")
    print(f"  Original avg:  {orig_train['full_text'].str.len().mean():.0f} chars")
    print(f"  Enhanced avg:  {enh_train['full_text'].str.len().mean():.0f} chars")

    return {
        "original": {
            "train": orig_train,
            "dev": orig_dev,
            "test": orig_test,
            "preprocessor": original_prep
        },
        "enhanced": {
            "train": enh_train,
            "dev": enh_dev,
            "test": enh_test,
            "preprocessor": enhanced_prep
        }
    }
