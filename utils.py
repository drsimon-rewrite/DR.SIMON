"""
Utility functions for DR.SIMON

This module contains helper functions used across the DR.SIMON pipeline.
"""

import json
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def load_video_features(feat_path: str) -> Optional[np.ndarray]:
    """
    Load video features from a numpy file.
    
    Args:
        feat_path: Path to the .npy feature file
        
    Returns:
        Feature array or None if file doesn't exist
    """
    feat_path = Path(feat_path)
    if feat_path.exists():
        return np.load(feat_path)
    return None


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to a JSON file.
    
    Args:
        results: Dictionary containing results to save
        output_path: Path to save the results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_json_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        data_path: Path to the JSON file
        
    Returns:
        List of data items
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        return [data]
    return data


def compute_iou(span1: Tuple[float, float], span2: Tuple[float, float]) -> float:
    """
    Compute Intersection over Union between two time spans.
    
    Args:
        span1: First time span (start, end)
        span2: Second time span (start, end)
        
    Returns:
        IoU value between 0 and 1
    """
    start1, end1 = span1
    start2, end2 = span2
    
    # Intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def format_time(seconds: float) -> str:
    """
    Format time in seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def print_evaluation_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title for the results
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:>8}: {value:.4f}")
        else:
            print(f"{metric:>8}: {value}")


def visualize_temporal_predictions(predictions: List[Tuple[float, float]],
                                 ground_truth: List[Tuple[float, float]],
                                 video_durations: List[float],
                                 save_path: Optional[str] = None):
    """
    Visualize temporal predictions vs ground truth.
    
    Args:
        predictions: List of predicted spans
        ground_truth: List of ground truth spans
        video_durations: List of video durations
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(len(predictions), 1, figsize=(12, 2*len(predictions)))
    if len(predictions) == 1:
        axes = [axes]
    
    for i, (pred, gt, duration) in enumerate(zip(predictions, ground_truth, video_durations)):
        ax = axes[i]
        
        # Draw timeline
        ax.barh(0, duration, height=0.3, color='lightgray', alpha=0.5, label='Video')
        
        # Draw ground truth
        ax.barh(0.1, gt[1] - gt[0], left=gt[0], height=0.15, 
                color='green', alpha=0.7, label='Ground Truth')
        
        # Draw prediction
        ax.barh(-0.1, pred[1] - pred[0], left=pred[0], height=0.15, 
                color='red', alpha=0.7, label='Prediction')
        
        # Compute IoU
        iou = compute_iou(pred, gt)
        
        ax.set_xlim(0, duration)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Video {i+1} - IoU: {iou:.3f}')
        ax.legend()
        
        # Remove y-axis
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def create_attention_heatmap(attention_weights: np.ndarray,
                           tokens: List[str],
                           save_path: Optional[str] = None):
    """
    Create a heatmap visualization of attention weights.
    
    Args:
        attention_weights: Attention weight matrix [num_tokens, num_tokens]
        tokens: List of token strings
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(attention_weights, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                cbar=True,
                square=True)
    
    plt.title('Cross-Modal Attention Weights')
    plt.xlabel('Video Tokens')
    plt.ylabel('Text Tokens')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")
    
    plt.show()


def split_text_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple rules.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    import re
    
    # Split on sentence-ending punctuation, but avoid splitting on decimals
    sentence_endings = re.compile(r'(?<!\d)\.(?!\d)|[!?]+')
    sentences = sentence_endings.split(text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 3:  # Filter very short fragments
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings.
    
    Args:
        embeddings: Input embeddings [N, D]
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    return embeddings / norms


def cosine_similarity_matrix(embeddings1: np.ndarray, 
                           embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [M, D]
        
    Returns:
        Similarity matrix [N, M]
    """
    # Normalize embeddings
    emb1_norm = normalize_embeddings(embeddings1)
    emb2_norm = normalize_embeddings(embeddings2)
    
    # Compute cosine similarity
    return np.dot(emb1_norm, emb2_norm.T)


def filter_events_by_duration(events: List[Dict[str, Any]], 
                            min_duration: float = 1.0,
                            max_duration: float = 300.0) -> List[Dict[str, Any]]:
    """
    Filter events by duration constraints.
    
    Args:
        events: List of event dictionaries
        min_duration: Minimum event duration in seconds
        max_duration: Maximum event duration in seconds
        
    Returns:
        Filtered list of events
    """
    filtered_events = []
    
    for event in events:
        start_time = event.get('start_time', 0.0)
        end_time = event.get('end_time', 0.0)
        duration = end_time - start_time
        
        if min_duration <= duration <= max_duration:
            filtered_events.append(event)
    
    return filtered_events


def merge_overlapping_spans(spans: List[Tuple[float, float]], 
                          min_gap: float = 0.5) -> List[Tuple[float, float]]:
    """
    Merge overlapping or closely spaced time spans.
    
    Args:
        spans: List of (start, end) tuples
        min_gap: Minimum gap to keep spans separate
        
    Returns:
        List of merged spans
    """
    if not spans:
        return []
    
    # Sort spans by start time
    sorted_spans = sorted(spans, key=lambda x: x[0])
    
    merged = [sorted_spans[0]]
    
    for current_start, current_end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        
        # Check if spans should be merged
        if current_start <= last_end + min_gap:
            # Merge spans
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # Add as new span
            merged.append((current_start, current_end))
    
    return merged


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
    def update(self, message: str = ""):
        self.current_step += 1
        progress = self.current_step / self.total_steps * 100
        print(f"\r{self.description}: {progress:.1f}% {message}", end="", flush=True)
        
        if self.current_step == self.total_steps:
            print()  # New line when complete
    
    def finish(self, message: str = "Complete"):
        print(f"\r{self.description}: 100.0% {message}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['n_segments', 'overlap', 'similarity_threshold']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return False
    
    # Validate ranges
    if not (1 <= config['n_segments'] <= 50):
        print("n_segments must be between 1 and 50")
        return False
    
    if not (0.0 <= config['overlap'] < 1.0):
        print("overlap must be between 0.0 and 1.0")
        return False
    
    if not (0.0 <= config['similarity_threshold'] <= 1.0):
        print("similarity_threshold must be between 0.0 and 1.0")
        return False
    
    return True 