"""
Data utilities for makemore character-level language models.

This module provides functions to:
- Load names from text files
- Build character vocabularies
- Create train/val/test splits
- Convert names to integer sequences
- Create PyTorch datasets
"""

import torch
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter


class NameDataset:
    """
    Character-level dataset for name generation.
    
    Handles vocabulary building, train/val/test splits, and encoding/decoding.
    """
    
    def __init__(self, file_path: str, train_split: float = 0.8, val_split: float = 0.1):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to the text file containing names (one per line)
            train_split: Proportion of data for training (default: 0.8)
            val_split: Proportion of data for validation (default: 0.1)
                      Test split will be 1 - train_split - val_split
        """
        self.file_path = Path(file_path)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split
        
        # Load and process data
        self.names = self._load_names()
        print(f"Loaded {len(self.names):,} names from {self.file_path.name}")
        
        # Build vocabulary
        self.chars, self.stoi, self.itos = self._build_vocab()
        self.vocab_size = len(self.chars)
        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"   Characters: {''.join(self.chars[1:])}")  # Skip special token
        
        # Create splits
        self.train_names, self.val_names, self.test_names = self._create_splits()
        print(f"Dataset splits:")
        print(f"   Train: {len(self.train_names):,} names ({self.train_split*100:.0f}%)")
        print(f"   Val:   {len(self.val_names):,} names ({self.val_split*100:.0f}%)")
        print(f"   Test:  {len(self.test_names):,} names ({self.test_split*100:.0f}%)")
    
    def _load_names(self) -> List[str]:
        """Load names from file and clean them."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            names = f.read().splitlines()
        
        # Remove empty lines and strip whitespace
        names = [name.strip() for name in names if name.strip()]
        
        # Convert to lowercase for consistency
        names = [name.lower() for name in names]
        
        return names
    
    def _build_vocab(self) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        """
        Build character vocabulary from all names.
        
        Returns:
            chars: List of unique characters (sorted)
            stoi: String to integer mapping
            itos: Integer to string mapping
        """
        # Get all unique characters
        all_chars = set()
        for name in self.names:
            all_chars.update(name)
        
        # Sort characters for consistency
        chars = sorted(list(all_chars))
        
        # Add special token at the beginning for start/end of sequence
        chars = ['.'] + chars
        
        # Create mappings
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        return chars, stoi, itos
    
    def _create_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split dataset into train/val/test sets.
        
        Uses a fixed random seed for reproducibility.
        """
        # Shuffle with fixed seed for reproducibility
        random.seed(2147483647)
        names_shuffled = self.names.copy()
        random.shuffle(names_shuffled)
        
        # Calculate split indices
        n = len(names_shuffled)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)
        
        # Split the data
        train_names = names_shuffled[:train_end]
        val_names = names_shuffled[train_end:val_end]
        test_names = names_shuffled[val_end:]
        
        return train_names, val_names, test_names
    
    def encode(self, name: str) -> List[int]:
        """
        Convert a name (string) to a list of integers.
        
        Args:
            name: The name to encode
            
        Returns:
            List of integer indices
        """
        return [self.stoi[ch] for ch in name]
    
    def decode(self, indices: List[int]) -> str:
        """
        Convert a list of integers back to a name (string).
        
        Args:
            indices: List of integer indices
            
        Returns:
            Decoded name as string
        """
        return ''.join([self.itos[i] for i in indices])
    
    def get_char_counts(self, split: str = 'train') -> Counter:
        """
        Get character frequency counts for a given split.
        
        Args:
            split: One of 'train', 'val', 'test', or 'all'
            
        Returns:
            Counter object with character frequencies
        """
        if split == 'train':
            names = self.train_names
        elif split == 'val':
            names = self.val_names
        elif split == 'test':
            names = self.test_names
        elif split == 'all':
            names = self.names
        else:
            raise ValueError(f"Unknown split: {split}")
        
        counter = Counter()
        for name in names:
            counter.update(name)
        
        return counter
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_names': len(self.names),
            'train_names': len(self.train_names),
            'val_names': len(self.val_names),
            'test_names': len(self.test_names),
            'vocab_size': self.vocab_size,
            'min_length': min(len(name) for name in self.names),
            'max_length': max(len(name) for name in self.names),
            'avg_length': sum(len(name) for name in self.names) / len(self.names),
        }
        return stats
    
    def print_stats(self):
        """Print dataset statistics in a readable format."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total names:    {stats['total_names']:,}")
        print(f"Train names:    {stats['train_names']:,}")
        print(f"Val names:      {stats['val_names']:,}")
        print(f"Test names:     {stats['test_names']:,}")
        print(f"Vocabulary:     {stats['vocab_size']} characters")
        print(f"Name length:    min={stats['min_length']}, "
              f"max={stats['max_length']}, avg={stats['avg_length']:.1f}")
        print("="*50 + "\n")


def load_dataset(file_path: str, train_split: float = 0.8, val_split: float = 0.1) -> NameDataset:
    """
    Convenience function to load and prepare the dataset.
    
    Args:
        file_path: Path to the names text file
        train_split: Proportion for training (default: 0.8)
        val_split: Proportion for validation (default: 0.1)
        
    Returns:
        NameDataset object with train/val/test splits
    """
    dataset = NameDataset(file_path, train_split, val_split)
    dataset.print_stats()
    return dataset


# Example usage
if __name__ == "__main__":
    # Test the data utilities
    dataset = load_dataset("../data/processed/names_group3.txt")
    
    # Test encoding/decoding
    test_name = "emma"
    encoded = dataset.encode(test_name)
    decoded = dataset.decode(encoded)
    
    print(f"Original: {test_name}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    
    # Show some example names from each split
    print("\nExample names from each split:")
    print(f"Train: {dataset.train_names[:5]}")
    print(f"Val:   {dataset.val_names[:5]}")
    print(f"Test:  {dataset.test_names[:5]}")
