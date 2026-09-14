# -*- coding: utf-8 -*-
"""
Evaluation metrics for symbolic counting agents.
Calculates accuracy, compression ratio, robustness, and other metrics.
"""
from typing import List, Dict, Tuple, Any
import math

class EvaluationMetrics:
    """Compute various evaluation metrics for Agent-U performance."""
    
    @staticmethod
    def accuracy(N: int, N_hat: int) -> float:
        """
        Calculate counting accuracy.
        Returns 1.0 for perfect match, decreases with absolute error.
        """
        if N == N_hat:
            return 1.0
        
        # Avoid division by zero
        max_value = max(N, N_hat, 1)
        absolute_error = abs(N - N_hat)
        return max(0.0, 1.0 - (absolute_error / max_value))
    
    @staticmethod
    def compression_ratio(original_length: int, compressed_length: int) -> float:
        """
        Calculate compression ratio: original_length / compressed_length.
        Higher values indicate better compression.
        """
        if compressed_length <= 0:
            return 1.0
        return original_length / compressed_length
    
    @staticmethod
    def compression_efficiency(original_length: int, rle_pairs: List[Tuple[str, int]]) -> float:
        """
        Calculate compression efficiency for RLE encoding.
        Each RLE pair (symbol, count) takes 2 storage units.
        """
        if not rle_pairs:
            return 1.0
        
        compressed_length = len(rle_pairs) * 2  # Each pair takes 2 units
        return EvaluationMetrics.compression_ratio(original_length, compressed_length)
    
    @staticmethod
    def symbol_diversity(rle_pairs: List[Tuple[str, int]]) -> float:
        """
        Calculate symbol diversity in encoded sequence.
        Higher diversity indicates more efficient dictionary usage.
        """
        if not rle_pairs:
            return 0.0
            
        unique_symbols = set(symbol for symbol, _ in rle_pairs)
        total_pairs = len(rle_pairs)
        
        return len(unique_symbols) / total_pairs if total_pairs > 0 else 0.0
    
    @staticmethod
    def dictionary_utilization(dictionary: Dict[str, str], capacity: int) -> float:
        """
        Calculate how efficiently the dictionary is being used.
        Returns percentage of dictionary slots used.
        """
        if capacity <= 0:
            return 0.0
        return len(dictionary) / capacity
    
    @staticmethod
    def entropy(rle_pairs: List[Tuple[str, int]]) -> float:
        """
        Calculate Shannon entropy of symbol distribution.
        Higher entropy indicates more uniform symbol usage.
        """
        if not rle_pairs:
            return 0.0
            
        # Count frequency of each symbol
        symbol_counts = {}
        total_count = 0
        
        for symbol, count in rle_pairs:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + count
            total_count += count
        
        if total_count <= 0:
            return 0.0
            
        # Calculate entropy
        entropy = 0.0
        for count in symbol_counts.values():
            probability = count / total_count
            if probability > 0:
                entropy -= probability * math.log2(probability)
                
        return entropy
    
    @staticmethod
    def run_length_efficiency(rle_pairs: List[Tuple[str, int]]) -> float:
        """
        Calculate average run length - higher values indicate better RLE compression.
        """
        if not rle_pairs:
            return 0.0
            
        total_run_length = sum(count for _, count in rle_pairs)
        num_runs = len(rle_pairs)
        
        return total_run_length / num_runs if num_runs > 0 else 0.0
    
    @classmethod
    def comprehensive_evaluation(cls, agent, N: int, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of an agent's performance.
        
        Args:
            agent: AgentU instance
            N: True count
            episode_data: Episode information
            
        Returns:
            Dictionary with all computed metrics
        """
        # Get encoding and decoding results
        rle_encoding = agent.encode()
        N_hat = agent.decode(rle_encoding)
        dictionary = agent.get_dictionary()
        
        # Calculate metrics
        accuracy = cls.accuracy(N, N_hat)
        
        original_length = len(agent._obs) if agent._obs else 0
        compression_eff = cls.compression_efficiency(original_length, rle_encoding)
        
        symbol_div = cls.symbol_diversity(rle_encoding)
        dict_util = cls.dictionary_utilization(dictionary["symbols"], dictionary["capacity"])
        entropy = cls.entropy(rle_encoding)
        run_length_eff = cls.run_length_efficiency(rle_encoding)
        
        # Memory usage
        memory_usage = {
            "D": agent.wm.D,
            "R": agent.wm.R,
            "K": agent.wm.K,
            "D_utilization": dict_util
        }
        
        return {
            "accuracy": accuracy,
            "compression_efficiency": compression_eff,
            "symbol_diversity": symbol_div,
            "dictionary_utilization": dict_util,
            "entropy": entropy,
            "run_length_efficiency": run_length_eff,
            "memory_usage": memory_usage,
            "N": N,
            "N_hat": N_hat,
            "rle_pairs": len(rle_encoding),
            "original_length": original_length,
            "dictionary_size": len(dictionary["symbols"]),
            "overflow_used": dictionary["overflow_used"]
        }