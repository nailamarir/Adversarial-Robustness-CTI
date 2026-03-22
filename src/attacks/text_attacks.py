"""
Text-Level Adversarial Attacks Module
Various text perturbation strategies for robustness evaluation
"""

import random
import string
from abc import ABC, abstractmethod
from typing import List, Optional, Set
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextAttacker(ABC):
    """Base class for text-level adversarial attacks"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    @abstractmethod
    def attack(self, text: str) -> str:
        """Apply attack to text"""
        pass

    def __call__(self, text: str) -> str:
        return self.attack(text)


class SynonymAttack(TextAttacker):
    """Replace words with synonyms using WordNet"""

    def __init__(
        self,
        num_replacements: int = 3,
        min_word_length: int = 4,
        seed: int = 42
    ):
        super().__init__(seed)
        self.num_replacements = num_replacements
        self.min_word_length = min_word_length

        # Words to avoid replacing
        self.stopwords: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall"
        }

    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)

    def attack(self, text: str) -> str:
        """Replace random words with synonyms"""
        words = text.split()

        # Find replaceable words
        replaceable = [
            (i, w) for i, w in enumerate(words)
            if len(w) >= self.min_word_length
            and w.isalpha()
            and w.lower() not in self.stopwords
        ]

        if not replaceable:
            return text

        # Select words to replace
        num_to_replace = min(self.num_replacements, len(replaceable))
        selected = random.sample(replaceable, num_to_replace)

        # Apply replacements
        for idx, word in selected:
            synonyms = self.get_synonyms(word.lower())
            if synonyms:
                replacement = random.choice(synonyms)
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement

        return " ".join(words)


class CharacterSwapAttack(TextAttacker):
    """Swap adjacent characters to create typos"""

    def __init__(
        self,
        num_swaps: int = 2,
        seed: int = 42
    ):
        super().__init__(seed)
        self.num_swaps = num_swaps

    def attack(self, text: str) -> str:
        """Swap adjacent characters in random positions"""
        text_list = list(text)

        # Find valid swap positions (between two letters)
        valid_positions = [
            i for i in range(len(text_list) - 1)
            if text_list[i].isalpha() and text_list[i + 1].isalpha()
        ]

        if not valid_positions:
            return text

        # Perform swaps
        num_to_swap = min(self.num_swaps, len(valid_positions))
        positions = random.sample(valid_positions, num_to_swap)

        for pos in positions:
            text_list[pos], text_list[pos + 1] = text_list[pos + 1], text_list[pos]

        return "".join(text_list)


class HomoglyphAttack(TextAttacker):
    """Replace characters with visually similar Unicode characters"""

    def __init__(
        self,
        num_replacements: int = 3,
        seed: int = 42
    ):
        super().__init__(seed)
        self.num_replacements = num_replacements

        # Homoglyph mappings (ASCII -> Unicode lookalikes)
        self.homoglyphs = {
            'a': ['а', 'ɑ', 'α'],  # Cyrillic, Latin, Greek
            'e': ['е', 'ε', 'ё'],
            'o': ['о', 'ο', 'ö'],
            'i': ['і', 'ι', 'ί'],
            'c': ['с', 'ϲ'],
            'p': ['р', 'ρ'],
            's': ['ѕ', 'ś'],
            'x': ['х', 'χ'],
            'y': ['у', 'ý'],
            'n': ['η', 'ñ'],
        }

    def attack(self, text: str) -> str:
        """Replace characters with homoglyphs"""
        text_list = list(text)

        # Find replaceable positions
        replaceable = [
            i for i, c in enumerate(text_list)
            if c.lower() in self.homoglyphs
        ]

        if not replaceable:
            return text

        num_to_replace = min(self.num_replacements, len(replaceable))
        positions = random.sample(replaceable, num_to_replace)

        for pos in positions:
            char = text_list[pos].lower()
            replacement = random.choice(self.homoglyphs[char])
            text_list[pos] = replacement

        return "".join(text_list)


class KeyboardTypoAttack(TextAttacker):
    """Introduce typos based on keyboard proximity"""

    def __init__(
        self,
        num_typos: int = 2,
        seed: int = 42
    ):
        super().__init__(seed)
        self.num_typos = num_typos

        # QWERTY keyboard neighbors
        self.keyboard_neighbors = {
            'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf',
            't': 'ryfg', 'y': 'tugh', 'u': 'yihj', 'i': 'uojk',
            'o': 'iplk', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz',
            'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb',
            'j': 'huiknm', 'k': 'jiolm', 'l': 'kop', 'z': 'asx',
            'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
            'n': 'bhjm', 'm': 'njk'
        }

    def attack(self, text: str) -> str:
        """Introduce keyboard proximity typos"""
        text_list = list(text)

        # Find replaceable positions
        replaceable = [
            i for i, c in enumerate(text_list)
            if c.lower() in self.keyboard_neighbors
        ]

        if not replaceable:
            return text

        num_to_replace = min(self.num_typos, len(replaceable))
        positions = random.sample(replaceable, num_to_replace)

        for pos in positions:
            char = text_list[pos].lower()
            neighbors = self.keyboard_neighbors[char]
            replacement = random.choice(neighbors)

            # Preserve case
            if text_list[pos].isupper():
                replacement = replacement.upper()
            text_list[pos] = replacement

        return "".join(text_list)


class CharacterDeletionAttack(TextAttacker):
    """Randomly delete characters from words"""

    def __init__(
        self,
        num_deletions: int = 2,
        min_word_length: int = 5,
        seed: int = 42
    ):
        super().__init__(seed)
        self.num_deletions = num_deletions
        self.min_word_length = min_word_length

    def attack(self, text: str) -> str:
        """Delete random characters from words"""
        words = text.split()
        result_words = []

        deletions_made = 0
        for word in words:
            if (
                deletions_made < self.num_deletions
                and len(word) >= self.min_word_length
                and word.isalpha()
            ):
                # Delete a random character (not first or last)
                if len(word) > 2:
                    pos = random.randint(1, len(word) - 2)
                    word = word[:pos] + word[pos + 1:]
                    deletions_made += 1
            result_words.append(word)

        return " ".join(result_words)


class CharacterInsertionAttack(TextAttacker):
    """Randomly insert characters into words"""

    def __init__(
        self,
        num_insertions: int = 2,
        min_word_length: int = 4,
        seed: int = 42
    ):
        super().__init__(seed)
        self.num_insertions = num_insertions
        self.min_word_length = min_word_length

    def attack(self, text: str) -> str:
        """Insert random characters into words"""
        words = text.split()
        result_words = []

        insertions_made = 0
        for word in words:
            if (
                insertions_made < self.num_insertions
                and len(word) >= self.min_word_length
                and word.isalpha()
            ):
                # Insert a random letter
                pos = random.randint(1, len(word) - 1)
                char = random.choice(string.ascii_lowercase)
                word = word[:pos] + char + word[pos:]
                insertions_made += 1
            result_words.append(word)

        return " ".join(result_words)


class CombinedAttack(TextAttacker):
    """Combine multiple attack strategies"""

    def __init__(
        self,
        attacks: Optional[List[TextAttacker]] = None,
        seed: int = 42
    ):
        super().__init__(seed)

        if attacks is None:
            # Aggressive combination for meaningful perturbation
            self.attacks = [
                SynonymAttack(num_replacements=8, seed=seed),
                CharacterSwapAttack(num_swaps=5, seed=seed),
                KeyboardTypoAttack(num_typos=4, seed=seed),
                HomoglyphAttack(num_replacements=5, seed=seed),
                CharacterDeletionAttack(num_deletions=3, seed=seed),
                CharacterInsertionAttack(num_insertions=3, seed=seed),
            ]
        else:
            self.attacks = attacks

    def attack(self, text: str) -> str:
        """Apply all attacks sequentially"""
        result = text
        for attack in self.attacks:
            result = attack.attack(result)
        return result


class RandomAttackSelector(TextAttacker):
    """Randomly select and apply one attack from a pool"""

    def __init__(
        self,
        attacks: Optional[List[TextAttacker]] = None,
        seed: int = 42
    ):
        super().__init__(seed)

        if attacks is None:
            self.attacks = [
                SynonymAttack(seed=seed),
                CharacterSwapAttack(seed=seed),
                KeyboardTypoAttack(seed=seed),
                HomoglyphAttack(seed=seed)
            ]
        else:
            self.attacks = attacks

    def attack(self, text: str) -> str:
        """Randomly select and apply one attack"""
        selected_attack = random.choice(self.attacks)
        return selected_attack.attack(text)


class BERTAttackSimulated(TextAttacker):
    """
    Simulated BERT-Attack: context-aware token replacement.
    Uses WordNet synonyms filtered by POS tags as a lightweight proxy
    for masked LM replacement (avoids loading a separate BERT model).
    More sophisticated than simple SynonymAttack — considers POS and
    replaces content words with higher probability.
    """

    def __init__(self, num_replacements: int = 5, seed: int = 42):
        super().__init__(seed)
        self.num_replacements = num_replacements

    def _get_pos_synonyms(self, word: str, pos_tag: str) -> List[str]:
        """Get synonyms matching the POS tag."""
        wn_pos = {'NN': wordnet.NOUN, 'VB': wordnet.VERB,
                   'JJ': wordnet.ADJ, 'RB': wordnet.ADV}.get(pos_tag[:2])
        if not wn_pos:
            return []
        synonyms = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    synonyms.add(name)
        return list(synonyms)

    def attack(self, text: str) -> str:
        """Replace content words with POS-aware synonyms."""
        try:
            tokens = word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
        except Exception:
            return text

        replacements = 0
        result = []
        # Prioritize content words (nouns, verbs, adjectives)
        content_indices = [i for i, (w, t) in enumerate(tagged)
                          if t[:2] in ('NN', 'VB', 'JJ', 'RB') and len(w) > 3]
        random.shuffle(content_indices)
        replace_set = set(content_indices[:self.num_replacements])

        for i, (word, tag) in enumerate(tagged):
            if i in replace_set:
                syns = self._get_pos_synonyms(word, tag)
                if syns:
                    result.append(random.choice(syns))
                    replacements += 1
                    continue
            result.append(word)

        return " ".join(result)


def create_attack_suite(seed: int = 42) -> dict:
    """Create a dictionary of all available attacks"""
    return {
        "synonym": SynonymAttack(seed=seed),
        "char_swap": CharacterSwapAttack(seed=seed),
        "homoglyph": HomoglyphAttack(seed=seed),
        "keyboard_typo": KeyboardTypoAttack(seed=seed),
        "char_deletion": CharacterDeletionAttack(seed=seed),
        "char_insertion": CharacterInsertionAttack(seed=seed),
        "combined": CombinedAttack(seed=seed),
        "random": RandomAttackSelector(seed=seed),
        "bert_attack": BERTAttackSimulated(seed=seed),
    }
