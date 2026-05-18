import numpy as np
from typing import Iterator, Optional
from .mobius import Mobius


class KleinianGroup:
    def __init__(self, generators: list, names: list = None):
        self.generators = generators
        n = len(generators)
        if names is None:
            names = [chr(ord('a') + i) for i in range(n)]
        self.names = names
        self._transforms = {}
        for name, gen in zip(names, generators):
            self._transforms[name] = gen
            self._transforms[name.upper()] = gen.inverse()

    def word_to_mobius(self, word: str) -> Mobius:
        result = Mobius.identity()
        for ch in word:
            result = result * self._transforms[ch]
        return result

    def _is_cancellation(self, word: str, letter: str) -> bool:
        if not word:
            return False
        last = word[-1]
        if last.islower() and letter == last.upper():
            return True
        if last.isupper() and letter == last.lower():
            return True
        return False

    def enumerate_words(self, max_depth: int) -> Iterator:
        from collections import deque
        yield ("", Mobius.identity())
        queue = deque()
        queue.append(("", Mobius.identity()))
        while queue:
            word, transform = queue.popleft()
            if len(word) >= max_depth:
                continue
            for letter, t in self._transforms.items():
                if self._is_cancellation(word, letter):
                    continue
                new_word = word + letter
                new_transform = transform * t
                yield (new_word, new_transform)
                queue.append((new_word, new_transform))

    def orbit(self, z0: complex, max_depth: int) -> np.ndarray:
        points = []
        for word, transform in self.enumerate_words(max_depth):
            try:
                w = transform(z0)
                if np.isfinite(np.abs(w)):
                    points.append(w)
            except Exception:
                pass
        return np.array(points, dtype=np.complex128)

    def jorgensen_number(self, A: Mobius, B: Mobius) -> float:
        commutator = A * B * A.inverse() * B.inverse()
        val = abs(A.trace() ** 2 - 4) + abs(commutator.trace() - 2)
        return float(val)
