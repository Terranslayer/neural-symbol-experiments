# -*- coding: utf-8 -*-
from .memory import WorkingMemory
from typing import List, Tuple, Dict, Any, Optional
import random

class AgentU:
    """Agent-U with learnable dictionary and RLE baseline strategy."""

    def __init__(self, K: int, seed: int | None = None):
        self.wm = WorkingMemory(K)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        self._obs = None
        self._dictionary = {}  # symbol -> unique_id mapping
        self._symbol_counter = 0  # for generating unique symbol IDs
        
    def allocate(self, D: int):
        """Allocate memory and initialize dictionary with D slots."""
        D_actual, R_actual = self.wm.allocate(D)
        # Reset dictionary when allocation changes
        self._dictionary = {}
        self._symbol_counter = 0
        return D_actual, R_actual

    def observe(self, stream):
        """Store observation stream without any numerical priors."""
        self._obs = list(stream) if stream is not None else []
        return True

    def _get_or_create_symbol(self, pattern: Any) -> str:
        """Get existing symbol for pattern or create new one if dictionary has space."""
        pattern_str = str(pattern)
        
        if pattern_str in self._dictionary:
            return self._dictionary[pattern_str]
        
        # Check if we have space in dictionary (D constraint)
        if len(self._dictionary) < self.wm.D:
            symbol_id = f"s{self._symbol_counter}"
            self._dictionary[pattern_str] = symbol_id
            self._symbol_counter += 1
            return symbol_id
        else:
            # Dictionary full - use fallback symbol
            return "s_overflow"

    def encode(self) -> List[Tuple[str, int]]:
        """
        RLE encoding: compress consecutive repeated observations.
        Returns list of (symbol, run_length) pairs.
        Respects D (dictionary size) constraint.
        """
        if not self._obs or self.wm.D is None:
            return []
        
        # RLE compression without numerical priors
        compressed = []
        if not self._obs:
            return compressed
            
        current_item = self._obs[0]
        run_length = 1
        
        for i in range(1, len(self._obs)):
            if self._obs[i] == current_item:
                run_length += 1
            else:
                # End of run - encode it
                symbol = self._get_or_create_symbol(current_item)
                compressed.append((symbol, run_length))
                current_item = self._obs[i]
                run_length = 1
        
        # Don't forget the last run
        symbol = self._get_or_create_symbol(current_item)
        compressed.append((symbol, run_length))
        
        return compressed

    def decode(self, S: List[Tuple[str, int]]) -> int:
        """
        Reconstruct total positive events from RLE-encoded symbol sequence.

        Uses the most-frequent non-None entry in the stored observation stream
        as the reference positive pattern, then sums run-lengths of symbols
        that map back to that pattern. Avoids hardcoded digit/numeral tests.
        """
        if not S or not self._obs:
            return 0

        freq: Dict[str, int] = {}
        for obs in self._obs:
            if obs is None:
                continue
            key = str(obs)
            freq[key] = freq.get(key, 0) + 1

        if not freq:
            return 0

        positive_pattern_str = max(freq, key=freq.get)
        positive_symbol = self._dictionary.get(positive_pattern_str)

        if positive_symbol is None:
            return 0

        total = 0
        for symbol, run_length in S:
            if symbol == positive_symbol:
                total += run_length
        return total

    def get_dictionary(self) -> Dict[str, Any]:
        """Return current dictionary state for inspection."""
        return {
            "symbols": self._dictionary.copy(),
            "size": len(self._dictionary),
            "capacity": self.wm.D,
            "overflow_used": "s_overflow" in [v for v in self._dictionary.values()]
        }

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio of current encoding."""
        if not self._obs:
            return 1.0
        
        encoded = self.encode()
        original_length = len(self._obs)
        # Each RLE pair takes 2 units of space (symbol + count)
        compressed_length = len(encoded) * 2
        
        return original_length / max(compressed_length, 1)
