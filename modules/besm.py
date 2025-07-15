"""
Boundary-aware Event Segmentation Module (BESM)

This module segments videos into semantically coherent events by:
1. Creating overlapping sliding windows across the video
2. Generating action captions for each window using VLM
3. Clustering similar actions to form representative events
4. Refining temporal boundaries of events
"""

import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import re


class BoundaryAwareEventSegmentationModule:
    """
    Boundary-aware Event Segmentation Module that divides videos into 
    semantically coherent action events.
    """
    
    def __init__(self,
                 n_segments: int = 10,
                 overlap: float = 0.5,
                 similarity_threshold: float = 0.99,
                 sentence_model: str = "all-MiniLM-L6-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the BESM module.
        
        Args:
            n_segments: Number of sliding windows to create
            overlap: Overlap ratio between adjacent windows
            similarity_threshold: Threshold for clustering similar actions
            sentence_model: Sentence transformer model name
            device: Device to run computations on
        """
        self.n_segments = n_segments
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Initialize sentence embedder for action clustering
        self.embedder = SentenceTransformer(sentence_model)
        self.embedder.to(device)
        
        # JSON parsing pattern
        self.json_pattern = re.compile(r"\[\{.*?\}\]", re.S)
    
    def make_windows(self, duration: float) -> List[Tuple[float, float]]:
        """
        Create overlapping sliding windows across video duration.
        
        Args:
            duration: Video duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples for each window
        """
        if self.n_segments <= 1:
            return [(0.0, duration)]
        
        # Calculate stride and window length
        stride = duration / self.n_segments
        window_length = stride * (1 + self.overlap)
        
        windows = []
        for i in range(self.n_segments):
            start_time = max(0.0, i * stride - self.overlap * stride)
            end_time = min(duration, start_time + window_length)
            windows.append((start_time, end_time))
        
        return windows
    
    def extract_window_features(self, 
                              video_features: np.ndarray,
                              fps: float,
                              window: Tuple[float, float]) -> np.ndarray:
        """
        Extract features for a specific time window.
        
        Args:
            video_features: Full video features [num_frames, feature_dim]
            fps: Frames per second
            window: (start_time, end_time) tuple
            
        Returns:
            Window features [window_frames, feature_dim]
        """
        start_time, end_time = window
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Ensure valid frame indices
        start_frame = max(0, start_frame)
        end_frame = min(len(video_features), end_frame)
        
        if start_frame >= end_frame:
            return video_features[start_frame:start_frame+1]
        
        return video_features[start_frame:end_frame]
    
    def parse_action_captions(self, caption_text: str) -> List[Dict]:
        """
        Parse action captions from VLM output.
        
        Args:
            caption_text: Raw text output from VLM
            
        Returns:
            List of action dictionaries with start/end times and captions
        """
        match = self.json_pattern.search(caption_text)
        if not match:
            return []
        
        try:
            actions = json.loads(match.group())
            return actions
        except json.JSONDecodeError:
            return []
    
    def generate_window_captions(self, 
                               window_features: np.ndarray,
                               vlm_model,
                               window_idx: int) -> List[Dict]:
        """
        Generate action captions for a single window using VLM.
        
        Args:
            window_features: Features for the window
            vlm_model: Vision-language model
            window_idx: Index of the window
            
        Returns:
            List of action captions with timestamps
        """
        if vlm_model is None:
            # Fallback: generate dummy actions
            return [{
                "start": 0,
                "end": 100,
                "caption": f"Action in window {window_idx}"
            }]
        
        # Generate captions using VLM
        prompt = "Describe the actions happening in this video segment with start and end times."
        raw_output = vlm_model.generate(window_features, prompt)
        
        # Parse the output
        actions = self.parse_action_captions(raw_output)
        
        return actions
    
    def convert_to_absolute_time(self, 
                               actions: List[Dict],
                               window: Tuple[float, float]) -> List[Dict]:
        """
        Convert relative timestamps to absolute timestamps.
        
        Args:
            actions: List of actions with relative timestamps (0-100)
            window: (start_time, end_time) of the window
            
        Returns:
            List of actions with absolute timestamps
        """
        start_time, end_time = window
        window_duration = end_time - start_time
        
        absolute_actions = []
        for action in actions:
            abs_action = action.copy()
            
            # Convert percentages to absolute times
            rel_start = action.get('start', 0) / 100.0
            rel_end = action.get('end', 100) / 100.0
            
            abs_action['start_time'] = start_time + rel_start * window_duration
            abs_action['end_time'] = start_time + rel_end * window_duration
            
            absolute_actions.append(abs_action)
        
        return absolute_actions
    
    def cluster_actions(self, all_actions: List[Dict]) -> List[Dict]:
        """
        Cluster similar actions across all windows.
        
        Args:
            all_actions: All actions from all windows
            
        Returns:
            List of representative actions for each cluster
        """
        if not all_actions:
            return []
        
        # Extract captions and embed them
        captions = [action.get('caption', '') for action in all_actions]
        embeddings = self.embedder.encode(captions)
        
        # Perform agglomerative clustering
        distance_threshold = 1 - self.similarity_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group actions by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(all_actions[i])
        
        # Create representative actions for each cluster
        representative_actions = []
        for cluster_actions in clusters.values():
            # Find earliest start and latest end
            start_times = [a['start_time'] for a in cluster_actions]
            end_times = [a['end_time'] for a in cluster_actions]
            
            # Choose caption closest to cluster centroid
            cluster_captions = [a.get('caption', '') for a in cluster_actions]
            cluster_embeddings = self.embedder.encode(cluster_captions)
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find closest caption to centroid
            similarities = np.dot(cluster_embeddings, centroid)
            best_idx = np.argmax(similarities)
            
            representative = {
                'start_time': min(start_times),
                'end_time': max(end_times),
                'caption': cluster_captions[best_idx],
                'cluster_size': len(cluster_actions)
            }
            
            representative_actions.append(representative)
        
        # Sort by start time
        representative_actions.sort(key=lambda x: x['start_time'])
        
        return representative_actions
    
    def segment_video(self, 
                     video_features: np.ndarray,
                     duration: float,
                     fps: float = 2.0,
                     vlm_model=None) -> List[Dict]:
        """
        Segment video into coherent events.
        
        Args:
            video_features: Video features [num_frames, feature_dim]
            duration: Video duration in seconds
            fps: Frames per second
            vlm_model: Vision-language model for captioning
            
        Returns:
            List of representative events with timestamps and captions
        """
        # Create sliding windows
        windows = self.make_windows(duration)
        
        # Process each window
        all_actions = []
        for i, window in enumerate(windows):
            # Extract features for this window
            window_features = self.extract_window_features(
                video_features, fps, window
            )
            
            # Generate captions for this window
            window_actions = self.generate_window_captions(
                window_features, vlm_model, i
            )
            
            # Convert to absolute timestamps
            absolute_actions = self.convert_to_absolute_time(
                window_actions, window
            )
            
            all_actions.extend(absolute_actions)
        
        # Cluster similar actions
        representative_events = self.cluster_actions(all_actions)
        
        return representative_events
    
    def process_video_batch(self, 
                          video_data: List[Dict],
                          vlm_model=None) -> Dict[str, List[Dict]]:
        """
        Process a batch of videos for event segmentation.
        
        Args:
            video_data: List of video dictionaries with features and metadata
            vlm_model: Vision-language model
            
        Returns:
            Dictionary mapping video IDs to their events
        """
        results = {}
        
        for video_item in video_data:
            video_id = video_item.get('video_id', 'unknown')
            video_features = video_item.get('features')
            duration = video_item.get('duration', 0.0)
            fps = video_item.get('fps', 2.0)
            
            if video_features is None:
                continue
            
            # Segment this video
            events = self.segment_video(
                video_features, duration, fps, vlm_model
            )
            
            results[video_id] = events
        
        return results 