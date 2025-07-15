"""
Query-Event Matching Module (QEM)

This module matches rewritten queries to video events by:
1. Computing similarity between query and event embeddings
2. Ranking events by relevance scores
3. Clustering temporally coherent high-scoring events
4. Selecting the best temporal span using density-weighted scoring
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer, util


class QueryEventMatchingModule:
    """
    Query-Event Matching Module that aligns rewritten queries with 
    segmented video events to predict temporal boundaries.
    """
    
    def __init__(self,
                 sentence_model: str = "all-MiniLM-L6-v2",
                 top_k: int = 7,
                 gap_threshold: float = 1.5,
                 density_weight: float = 0.1,
                 iou_threshold: float = 0.2,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the QEM module.
        
        Args:
            sentence_model: Sentence transformer model name
            top_k: Number of top-scoring events to consider
            gap_threshold: Maximum gap between events to merge (seconds)
            density_weight: Weight for density bonus in cluster scoring
            iou_threshold: IoU threshold for compact block detection
            device: Device to run computations on
        """
        self.top_k = top_k
        self.gap_threshold = gap_threshold
        self.density_weight = density_weight
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Initialize sentence embedder
        self.embedder = SentenceTransformer(sentence_model)
        self.embedder.to(device)
    
    def compute_similarity_scores(self, 
                                query: str, 
                                events: List[Dict]) -> List[float]:
        """
        Compute cosine similarity between query and event captions.
        
        Args:
            query: Rewritten query text
            events: List of event dictionaries with captions
            
        Returns:
            List of similarity scores for each event
        """
        if not events:
            return []
        
        # Get query embedding
        query_embedding = self.embedder.encode([query], convert_to_tensor=True)
        
        # Get event caption embeddings
        captions = [event.get('caption', '') for event in events]
        event_embeddings = self.embedder.encode(captions, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = util.cos_sim(query_embedding, event_embeddings)[0]
        
        return similarities.cpu().numpy().tolist()
    
    def compute_iou(self, span1: Tuple[float, float], span2: Tuple[float, float]) -> float:
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
    
    def is_compact_block(self, events: List[Dict]) -> bool:
        """
        Check if top-3 events form a compact temporal block.
        
        Args:
            events: List of top-scoring events (assumed to be top-3)
            
        Returns:
            True if events form a compact block
        """
        if len(events) < 2:
            return True
        
        # Check pairwise IoU and temporal gaps
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1, event2 = events[i], events[j]
                
                span1 = (event1['start_time'], event1['end_time'])
                span2 = (event2['start_time'], event2['end_time'])
                
                # Check IoU
                iou = self.compute_iou(span1, span2)
                if iou > self.iou_threshold:
                    continue
                
                # Check temporal gap
                gap = min(
                    abs(event1['start_time'] - event2['end_time']),
                    abs(event2['start_time'] - event1['end_time'])
                )
                
                if gap <= self.gap_threshold:
                    continue
                
                # If neither IoU nor gap condition is met
                return False
        
        return True
    
    def merge_adjacent_events(self, events: List[Dict]) -> List[List[Dict]]:
        """
        Merge temporally adjacent events into clusters.
        
        Args:
            events: List of events sorted by start time
            
        Returns:
            List of event clusters
        """
        if not events:
            return []
        
        clusters = []
        current_cluster = [events[0]]
        
        for i in range(1, len(events)):
            prev_event = current_cluster[-1]
            curr_event = events[i]
            
            # Check temporal gap
            gap = curr_event['start_time'] - prev_event['end_time']
            
            if gap <= self.gap_threshold:
                # Merge with current cluster
                current_cluster.append(curr_event)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [curr_event]
        
        # Add the last cluster
        clusters.append(current_cluster)
        
        return clusters
    
    def compute_cluster_score(self, 
                            cluster: List[Dict], 
                            similarities: List[float]) -> float:
        """
        Compute density-weighted score for an event cluster.
        
        Args:
            cluster: List of events in the cluster
            similarities: Similarity scores for all events
            
        Returns:
            Weighted cluster score
        """
        if not cluster:
            return 0.0
        
        # Get similarity scores for events in this cluster
        cluster_scores = []
        for event in cluster:
            event_idx = event.get('index', 0)
            if event_idx < len(similarities):
                cluster_scores.append(similarities[event_idx])
        
        if not cluster_scores:
            return 0.0
        
        # Average similarity score
        avg_score = np.mean(cluster_scores)
        
        # Compute density bonus
        if len(cluster) == 1:
            density_bonus = 1.0
        else:
            # Compute span and average gap
            start_times = [e['start_time'] for e in cluster]
            end_times = [e['end_time'] for e in cluster]
            
            span = max(end_times) - min(start_times)
            avg_gap = span / len(cluster)
            
            # Density factor (higher for tightly packed clusters)
            density_factor = max(0.0, 1.0 - avg_gap / self.gap_threshold)
            density_bonus = 1.0 + self.density_weight * density_factor
        
        return avg_score * density_bonus
    
    def get_cluster_span(self, cluster: List[Dict]) -> Tuple[float, float]:
        """
        Get the temporal span of a cluster.
        
        Args:
            cluster: List of events in the cluster
            
        Returns:
            (start_time, end_time) tuple for the cluster
        """
        start_times = [event['start_time'] for event in cluster]
        end_times = [event['end_time'] for event in cluster]
        
        return (min(start_times), max(end_times))
    
    def match_query_to_events(self, 
                            query: str, 
                            events: List[Dict]) -> Tuple[float, float]:
        """
        Match a rewritten query to video events and predict temporal span.
        
        Args:
            query: Rewritten query text
            events: List of segmented video events
            
        Returns:
            Predicted temporal span (start_time, end_time)
        """
        if not events:
            return (0.0, 0.0)
        
        # Add indices to events for tracking
        indexed_events = []
        for i, event in enumerate(events):
            indexed_event = event.copy()
            indexed_event['index'] = i
            indexed_events.append(indexed_event)
        
        # Compute similarity scores
        similarities = self.compute_similarity_scores(query, indexed_events)
        
        # Sort events by similarity score
        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        
        # Get top-k events
        top_events = [indexed_events[i] for i in sorted_indices[:self.top_k]]
        top_similarities = [similarities[i] for i in sorted_indices[:self.top_k]]
        
        # Fast path: check if top-3 form a compact block
        if len(top_events) >= 3:
            top_3_events = top_events[:3]
            if self.is_compact_block(top_3_events):
                return self.get_cluster_span(top_3_events)
        
        # Sort top events by start time for clustering
        top_events.sort(key=lambda x: x['start_time'])
        
        # Merge adjacent events into clusters
        clusters = self.merge_adjacent_events(top_events)
        
        # Score each cluster
        best_score = -1.0
        best_cluster = None
        
        for cluster in clusters:
            score = self.compute_cluster_score(cluster, similarities)
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        # Return span of best cluster
        if best_cluster:
            return self.get_cluster_span(best_cluster)
        
        # Fallback: return span of top event
        if top_events:
            top_event = top_events[0]
            return (top_event['start_time'], top_event['end_time'])
        
        return (0.0, 0.0)
    
    def process_batch(self, 
                     queries: List[str], 
                     video_events: Dict[str, List[Dict]],
                     video_ids: List[str]) -> List[Tuple[float, float]]:
        """
        Process a batch of queries for temporal grounding.
        
        Args:
            queries: List of rewritten queries
            video_events: Dictionary mapping video IDs to their events
            video_ids: List of video IDs corresponding to queries
            
        Returns:
            List of predicted temporal spans
        """
        predictions = []
        
        for i, query in enumerate(queries):
            video_id = video_ids[i] if i < len(video_ids) else 'unknown'
            events = video_events.get(video_id, [])
            
            # Match query to events
            predicted_span = self.match_query_to_events(query, events)
            predictions.append(predicted_span)
        
        return predictions
    
    def evaluate_predictions(self, 
                           predictions: List[Tuple[float, float]],
                           ground_truth: List[Tuple[float, float]],
                           iou_thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth using IoU metrics.
        
        Args:
            predictions: List of predicted spans
            ground_truth: List of ground truth spans
            iou_thresholds: IoU thresholds for recall computation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Compute IoUs
        ious = []
        for pred, gt in zip(predictions, ground_truth):
            iou = self.compute_iou(pred, gt)
            ious.append(iou)
        
        # Compute metrics
        metrics = {
            'mIoU': np.mean(ious)
        }
        
        # Compute recall at different IoU thresholds
        for threshold in iou_thresholds:
            recall = np.mean([iou >= threshold for iou in ious])
            metrics[f'R@{threshold}'] = recall
        
        return metrics 