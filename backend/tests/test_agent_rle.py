# -*- coding: utf-8 -*-
"""
Tests for Agent-U RLE encoding and learnable dictionary functionality.
"""
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.agent_u import AgentU
from backend.core.evaluation import EvaluationMetrics

class TestAgentURLE:
    """Test RLE encoding and dictionary learning in Agent-U."""
    
    def test_basic_allocation_constraint(self):
        """Test that D+R=K constraint is enforced."""
        agent = AgentU(K=6, seed=42)
        
        # Valid allocation
        D, R = agent.allocate(D=3)
        assert D == 3
        assert R == 3
        assert D + R == agent.wm.K
        
        # Test invalid allocations
        with pytest.raises(ValueError):
            agent.allocate(D=0)  # D must be > 0
            
        with pytest.raises(ValueError):
            agent.allocate(D=6)  # D must be < K
            
    def test_dictionary_creation(self):
        """Test dictionary creation with D constraint."""
        agent = AgentU(K=6, seed=42)
        agent.allocate(D=3)
        
        # Initially empty
        dict_state = agent.get_dictionary()
        assert dict_state["size"] == 0
        assert dict_state["capacity"] == 3
        
        # Add some patterns
        agent.observe([1, 1, None, -1, 1])
        encoding = agent.encode()
        
        dict_state = agent.get_dictionary()
        assert dict_state["size"] <= 3  # Should not exceed D
        assert len(dict_state["symbols"]) <= 3
        
    def test_rle_encoding_basic(self):
        """Test basic RLE encoding functionality."""
        agent = AgentU(K=6, seed=42)
        agent.allocate(D=4)
        
        # Simple repeated pattern
        agent.observe([1, 1, 1, None, None, 1])
        encoding = agent.encode()
        
        # Should have compressed runs
        assert len(encoding) <= 6  # Original length
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in encoding)
        assert all(isinstance(pair[1], int) and pair[1] > 0 for pair in encoding)
        
    def test_rle_encoding_compression(self):
        """Test that RLE actually compresses repeated sequences."""
        agent = AgentU(K=8, seed=42)
        agent.allocate(D=5)
        
        # Highly repetitive sequence
        agent.observe([1] * 20 + [None] * 15 + [1] * 10)
        encoding = agent.encode()
        
        # Should be much shorter than original
        assert len(encoding) <= 3  # Three distinct runs
        
        # Check compression ratio
        compression_ratio = agent.get_compression_ratio()
        assert compression_ratio > 1.0  # Should achieve compression
        
    def test_rle_decoding_accuracy(self):
        """Test RLE decoding produces accurate counts."""
        agent = AgentU(K=6, seed=42)
        agent.allocate(D=3)
        
        # Known sequence with specific food count
        food_events = [1, 1, None, 1, -1, 1, 1]
        expected_food_count = 5  # Count of '1' events
        
        agent.observe(food_events)
        encoding = agent.encode()
        decoded_count = agent.decode(encoding)
        
        assert decoded_count == expected_food_count
        
    def test_dictionary_overflow_handling(self):
        """Test behavior when dictionary exceeds D constraint."""
        agent = AgentU(K=5, seed=42)
        agent.allocate(D=2)  # Very small dictionary
        
        # Create sequence with more unique patterns than D allows
        agent.observe([1, 2, 3, 4, 5, 1, 2])
        encoding = agent.encode()
        
        dict_state = agent.get_dictionary()
        # Dictionary should not exceed capacity
        assert len(dict_state["symbols"]) <= 2
        
        # Should still produce valid encoding
        assert len(encoding) > 0
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in encoding)
        
    def test_empty_and_edge_cases(self):
        """Test edge cases like empty streams."""
        agent = AgentU(K=4, seed=42)
        agent.allocate(D=2)
        
        # Empty stream
        agent.observe([])
        encoding = agent.encode()
        assert encoding == []
        assert agent.decode(encoding) == 0
        
        # Single item
        agent.observe([1])
        encoding = agent.encode()
        assert len(encoding) == 1
        assert encoding[0][1] == 1  # Run length should be 1
        
    def test_no_numerical_priors(self):
        """Test that implementation avoids numerical priors."""
        agent = AgentU(K=6, seed=42)
        agent.allocate(D=3)
        
        # Use non-numerical patterns
        agent.observe(['food', 'food', 'neutral', 'interference', 'food'])
        encoding = agent.encode()
        
        # Should work with non-numerical data
        assert len(encoding) > 0
        dict_state = agent.get_dictionary()
        assert len(dict_state["symbols"]) <= 3
        
    def test_deterministic_behavior(self):
        """Test that same seed produces deterministic results."""
        # Create two identical agents
        agent1 = AgentU(K=6, seed=123)
        agent1.allocate(D=3)
        agent1.observe([1, 1, None, 1, -1])
        
        agent2 = AgentU(K=6, seed=123)
        agent2.allocate(D=3)
        agent2.observe([1, 1, None, 1, -1])
        
        encoding1 = agent1.encode()
        encoding2 = agent2.encode()
        
        assert encoding1 == encoding2
        assert agent1.get_dictionary()["symbols"] == agent2.get_dictionary()["symbols"]

class TestEvaluationMetrics:
    """Test evaluation metrics functionality."""
    
    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        # Perfect match
        assert EvaluationMetrics.accuracy(5, 5) == 1.0
        
        # Close match
        accuracy = EvaluationMetrics.accuracy(10, 9)
        assert 0.0 < accuracy < 1.0
        
        # Poor match
        accuracy = EvaluationMetrics.accuracy(1, 10)
        assert 0.0 <= accuracy < 0.5
        
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        # No compression
        assert EvaluationMetrics.compression_ratio(10, 10) == 1.0
        
        # Good compression
        assert EvaluationMetrics.compression_ratio(100, 10) == 10.0
        
        # Edge case
        assert EvaluationMetrics.compression_ratio(10, 0) == 1.0
        
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation function."""
        agent = AgentU(K=6, seed=42)
        agent.allocate(D=3)
        agent.observe([1, 1, None, 1, -1, 1])
        
        N = 4  # True count
        episode_data = {}
        
        metrics = EvaluationMetrics.comprehensive_evaluation(agent, N, episode_data)
        
        # Check that all expected metrics are present
        expected_keys = {
            "accuracy", "compression_efficiency", "symbol_diversity", 
            "dictionary_utilization", "entropy", "run_length_efficiency",
            "memory_usage", "N", "N_hat", "rle_pairs", "original_length"
        }
        
        assert all(key in metrics for key in expected_keys)
        assert isinstance(metrics["memory_usage"], dict)
        assert metrics["memory_usage"]["K"] == 6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])