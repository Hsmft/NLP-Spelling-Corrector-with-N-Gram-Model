from lm import SmoothedNGramLanguageModel
from typing import Iterable
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from nltk.stem.porter import PorterStemmer
import string
from collections import defaultdict
from functools import lru_cache
import math
from metaphone import doublemetaphone

class SimpleSpellingCorrector:
    # Initialize corrector with language model, set threshold tau, and setup keyboard/OCR maps.
    def __init__(self, lm: SmoothedNGramLanguageModel) -> None:
        self.lm = lm
        self.tau = 1.2
        self.stemmer = PorterStemmer()
        self.keyboard_map = {
            'a': 'qswz', 'b': 'vhn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
            'f': 'drtgv', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikm',
            'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'edfgt', 's': 'awedxz', 't': 'rfgy',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
            'z': 'asx'
        }
        self.ocr_map = {
            '0': 'o', '1': 'l', '2': 'z', '3': 'e', '4': 'a', '5': 's',
            '6': 'b', '7': 't', '8': 'b', '9': 'g', 'l': '1', 'o': '0',
            's': '5', 'b': '8'
        }
        self.phonetic_index = defaultdict(list)
        for w in lm.vocabulary:
            code = doublemetaphone(w)[0]
            if code:
                self.phonetic_index[code].append(w)

    # Generate candidate words for a word using keyboard, OCR, and phonetic errors.
    def _generate_candidates(self, word: str) -> Iterable[str]:
        w = word.lower()
        letters = string.ascii_lowercase
        candidates = set()

        # Replace digit characters with similar letters.
        for i, ch in enumerate(w):
            if ch.isdigit():
                for sub in self.ocr_map.get(ch, []):
                    cand = w[:i] + sub + w[i+1:]
                    candidates.add(cand)

        # Delete, insert, substitute, and swap characters.
        splits = [(w[:i], w[i:]) for i in range(len(w)+1)]
        for L, R in splits:
            if R:
                candidates.add(L + R[1:])  # delete
            for c in letters:
                candidates.add(L + c + R)  # insert
                if R:
                    candidates.add(L + c + R[1:])  # substitute
            if len(R) > 1:
                candidates.add(L + R[1] + R[0] + R[2:])  # swap

        # Keyboard mistakes.
        for i, ch in enumerate(w):
            for nb in self.keyboard_map.get(ch, ''):
                cand = w[:i] + nb + w[i+1:]
                candidates.add(cand)

        # OCR mistakes.
        for i, ch in enumerate(w):
            for sub in self.ocr_map.get(ch, ''):
                cand = w[:i] + sub + w[i+1:]
                candidates.add(cand)

        # Phonetic candidates.
        code = doublemetaphone(w)[0]
        if code:
            candidates.update(self.phonetic_index.get(code, []))

        # Filter candidates.
        max_ed = 5 if any(c.isdigit() for c in w) else 5
        valid = [c for c in candidates if self.stemmer.stem(c) in self.lm.vocabulary and
                 self.cached_edit_distance(c, w) <= max_ed and abs(len(c) - len(w)) <= 2]
        if not valid and self.stemmer.stem(w) in self.lm.vocabulary:
            valid = [w]
        return valid

    # Cache edit distance between two words for speed up.
    @lru_cache(maxsize=20000)
    def cached_edit_distance(self, word1, word2):
        return edit_distance(word1, word2)

    # Correct sentence by replace wrong words with best candidate based on n-gram probs.
    def correct(self, sentence: str) -> str:
        tokens = word_tokenize(sentence)
        n = self.lm.n
        stemmed_tokens = [self.stemmer.stem(t.lower()) for t in tokens]
        padded = ['<s>'] * (n - 1) + stemmed_tokens + ['</s>']
        corrected = tokens.copy()
        for i in range(n - 1, len(padded) - 1):
            orig = tokens[i - (n - 1)]
            orig_stemmed = padded[i]
            if orig_stemmed in self.lm.vocabulary:
                continue
            prev_ctx = tuple(padded[i - (n - 1):i])
            cands = self._generate_candidates(orig)
            orig_score = math.log(self.lm.get_ngram_prob(prev_ctx + (orig_stemmed,)) + 1e-12)
            best, best_score = orig, orig_score
            for c in cands:
                c_stemmed = self.stemmer.stem(c.lower())
                if c_stemmed not in self.lm.vocabulary:
                    continue
                score = math.log(self.lm.get_ngram_prob(prev_ctx + (c_stemmed,)) + 1e-12)
                edit_dist = self.cached_edit_distance(orig.lower(), c.lower())
                adjusted_score = score - edit_dist
                if adjusted_score > best_score:
                    best_score, best = adjusted_score, c
            if best != orig and math.exp(best_score - orig_score) > self.tau:
                idx = i - (n - 1)
                if tokens[idx].istitle():
                    best = best.capitalize()
                corrected[idx] = best
        return ' '.join(corrected)