#!/usr/bin/env python
"""
DR.SIMON Main Pipeline

This script demonstrates the complete DR.SIMON pipeline:
1. Load video data and medical queries
2. Rewrite queries using QRM (Query Rewriting Module)
3. Segment videos using BESM (Boundary-aware Event Segmentation Module)
4. Match queries to events using QEM (Query-Event Matching Module)
5. Evaluate results and save predictions
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import torch
from tqdm import tqdm

# Import DR.SIMON modules
from modules.qrm import QueryRewritingModule
from modules.besm import BoundaryAwareEventSegmentationModule
from modules.qem import QueryEventMatchingModule


class DrSimonPipeline:
    """
    Complete DR.SIMON pipeline that integrates QRM, BESM, and QEM modules.
    """
    
    def __init__(self, 
                 model_base: str = "lmsys/vicuna-7b-v1.5",
                 n_segments: int = 10,
                 overlap: float = 0.5,
                 similarity_threshold: float = 0.99,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the DR.SIMON pipeline.
        
        Args:
            model_base: Base model for VLM
            n_segments: Number of video segments for BESM
            overlap: Overlap ratio for sliding windows
            similarity_threshold: Threshold for clustering in BESM
            device: Device to run on
        """
        self.device = device
        self.model_base = model_base
        
        print(f"Initializing DR.SIMON on {device}...")
        
        # Initialize modules
        self.qrm = QueryRewritingModule(device=device)
        self.besm = BoundaryAwareEventSegmentationModule(
            n_segments=n_segments,
            overlap=overlap,
            similarity_threshold=similarity_threshold,
            device=device
        )
        self.qem = QueryEventMatchingModule(device=device)
        
        print("✓ All modules initialized successfully")
    
    def load_data(self, data_path: str, feat_folder: str) -> tuple:
        """
        Load video data and features.
        
        Args:
            data_path: Path to JSON file with queries and metadata
            feat_folder: Path to folder containing video features
            
        Returns:
            Tuple of (queries, video_features, video_metadata)
        """
        print(f"Loading data from {data_path}...")
        
        # Load query data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        queries = []
        video_metadata = {}
        video_features = {}
        
        for item in data:
            # Extract query information
            query_info = {
                'video_id': item.get('video_id', 'unknown'),
                'query': item.get('query', ''),
                'start_time': item.get('start_time', 0.0),
                'end_time': item.get('end_time', 0.0),
                'duration': item.get('duration', 0.0)
            }
            queries.append(query_info)
            
            # Load video features if available
            video_id = query_info['video_id']
            if video_id not in video_features:
                feat_path = Path(feat_folder) / f"{video_id}.npy"
                if feat_path.exists():
                    features = np.load(feat_path)
                    video_features[video_id] = features
                    video_metadata[video_id] = {
                        'duration': query_info['duration'],
                        'fps': 2.0  # Default FPS
                    }
        
        print(f"✓ Loaded {len(queries)} queries for {len(video_features)} videos")
        return queries, video_features, video_metadata
    
    def run_pipeline(self, 
                    queries: List[Dict], 
                    video_features: Dict[str, np.ndarray],
                    video_metadata: Dict[str, Dict]) -> List[Dict]:
        """
        Run the complete DR.SIMON pipeline.
        
        Args:
            queries: List of query dictionaries
            video_features: Dictionary of video features
            video_metadata: Dictionary of video metadata
            
        Returns:
            List of results with predictions
        """
        print("Running DR.SIMON pipeline...")
        
        # Step 1: Query Rewriting (QRM)
        print("Step 1/3: Rewriting queries...")
        rewritten_queries = []
        for query_info in tqdm(queries, desc="Rewriting queries"):
            original_query = query_info['query']
            video_id = query_info['video_id']
            
            # Get video context (simplified - in practice would use VLM)
            video_context = f"Video {video_id} context"
            
            # Rewrite query
            rewritten_query = self.qrm.rewrite_query(
                original_query, video_context, vlm_model=None
            )
            
            query_result = query_info.copy()
            query_result['rewritten_query'] = rewritten_query
            rewritten_queries.append(query_result)
        
        # Step 2: Video Event Segmentation (BESM)
        print("Step 2/3: Segmenting videos into events...")
        all_video_events = {}
        
        for video_id, features in tqdm(video_features.items(), desc="Segmenting videos"):
            metadata = video_metadata[video_id]
            duration = metadata['duration']
            fps = metadata.get('fps', 2.0)
            
            # Segment video into events
            events = self.besm.segment_video(
                features, duration, fps, vlm_model=None
            )
            all_video_events[video_id] = events
        
        # Step 3: Query-Event Matching (QEM)
        print("Step 3/3: Matching queries to events...")
        final_results = []
        
        for query_result in tqdm(rewritten_queries, desc="Matching queries"):
            video_id = query_result['video_id']
            rewritten_query = query_result['rewritten_query']
            
            # Get events for this video
            events = all_video_events.get(video_id, [])
            
            # Match query to events
            predicted_span = self.qem.match_query_to_events(rewritten_query, events)
            
            # Compile final result
            final_result = query_result.copy()
            final_result['predicted_start'] = predicted_span[0]
            final_result['predicted_end'] = predicted_span[1]
            final_result['num_events'] = len(events)
            
            final_results.append(final_result)
        
        print("✓ Pipeline completed successfully")
        return final_results
    
    def evaluate_results(self, results: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the pipeline results.
        
        Args:
            results: List of result dictionaries with predictions and ground truth
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating results...")
        
        predictions = []
        ground_truth = []
        
        for result in results:
            pred_span = (result['predicted_start'], result['predicted_end'])
            gt_span = (result['start_time'], result['end_time'])
            
            predictions.append(pred_span)
            ground_truth.append(gt_span)
        
        # Compute metrics using QEM
        metrics = self.qem.evaluate_predictions(predictions, ground_truth)
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics


def main():
    """Main function to run DR.SIMON pipeline."""
    parser = argparse.ArgumentParser(description="DR.SIMON: Domain-wise Rewrite for Segment-Informed Medical Oversight Network")
    
    parser.add_argument("--data_path", type=str, 
                       default="data/medical/rewrited_query/medvqa_test_grouped_rewrite_from_all_summarized.json",
                       help="Path to JSON file with queries and metadata")
    parser.add_argument("--feat_folder", type=str, 
                       default="data/medical/feature/test",
                       help="Path to folder containing video features (.npy files)")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5",
                       help="Base model for VLM")
    
    # VTimeLLM checkpoint paths
    parser.add_argument("--clip_path", type=str, 
                       default="checkpoints/clip/ViT-L-14.pt",
                       help="Path to CLIP checkpoint")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, 
                       default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin",
                       help="Path to VTimeLLM stage1 adapter")
    parser.add_argument("--stage2_path", type=str, 
                       default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2",
                       help="Path to VTimeLLM stage2 checkpoint")
    parser.add_argument("--stage3_path", type=str, 
                       default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3",
                       help="Path to VTimeLLM stage3 checkpoint")
    
    # DR.SIMON parameters
    parser.add_argument("--n_segments", type=int, default=10,
                       help="Number of video segments")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding windows")
    parser.add_argument("--similarity_threshold", type=float, default=0.99,
                       help="Similarity threshold for clustering")
    parser.add_argument("--output_path", type=str, default="results/predictions.json",
                       help="Path to save predictions")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Initialize pipeline
    pipeline = DrSimonPipeline(
        model_base=args.model_base,
        n_segments=args.n_segments,
        overlap=args.overlap,
        similarity_threshold=args.similarity_threshold,
        device=device
    )
    
    # Load data
    queries, video_features, video_metadata = pipeline.load_data(
        args.data_path, args.feat_folder
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(queries, video_features, video_metadata)
    
    # Evaluate results
    metrics = pipeline.evaluate_results(results)
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'results': results,
        'metrics': metrics,
        'config': {
            'model_base': args.model_base,
            'n_segments': args.n_segments,
            'overlap': args.overlap,
            'similarity_threshold': args.similarity_threshold
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main() 