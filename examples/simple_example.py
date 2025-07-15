
import sys
import os
import json
import numpy as np
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from modules.qrm import QueryRewritingModule
from modules.besm import BoundaryAwareEventSegmentationModule
from modules.qem import QueryEventMatchingModule
from utils import print_evaluation_metrics, compute_iou


def create_sample_data():
    """Create sample data for demonstration."""
    
    # Sample video features (dummy data)
    video_features = {
        'video_001': np.random.randn(600, 768),  # 5 minutes at 2fps
        'video_002': np.random.randn(480, 768),  # 4 minutes at 2fps
    }
    
    # Sample queries
    queries = [
        {
            'video_id': 'video_001',
            'query': 'How to perform chin tuck exercise?',
            'start_time': 45.0,
            'end_time': 78.0,
            'duration': 300.0
        },
        {
            'video_id': 'video_001', 
            'query': 'Demonstrate neck rotation technique',
            'start_time': 120.0,
            'end_time': 150.0,
            'duration': 300.0
        },
        {
            'video_id': 'video_002',
            'query': 'Show shoulder blade squeeze procedure',
            'start_time': 30.0,
            'end_time': 60.0,
            'duration': 240.0
        }
    ]
    
    # Video metadata
    video_metadata = {
        'video_001': {'duration': 300.0, 'fps': 2.0},
        'video_002': {'duration': 240.0, 'fps': 2.0}
    }
    
    return queries, video_features, video_metadata


def run_simple_example():
    """Run a simple example of the Dr.Simon pipeline."""
    
    print("🏥 DR.SIMON Simple Example")
    print("=" * 50)
    
    # Create sample data
    print(" Creating sample data...")
    queries, video_features, video_metadata = create_sample_data()
    print(f"✓ Created {len(queries)} queries for {len(video_features)} videos")
    
    # Initialize modules
    print("\nInitializing Dr.Simon modules...")
    qrm = QueryRewritingModule()
    besm = BoundaryAwareEventSegmentationModule(n_segments=5, overlap=0.5)
    qem = QueryEventMatchingModule()
    print("✓ All modules initialized")
    
    # Step 1: Query Rewriting
    print("\nStep 1: Rewriting medical queries...")
    rewritten_queries = []
    
    for i, query_info in enumerate(queries):
        original_query = query_info['query']
        print(f"  Query {i+1}: '{original_query}'")
        
        # Rewrite query (using fallback since no VLM available)
        rewritten_query = qrm.rewrite_query(original_query, "", vlm_model=None)
        print(f"  Rewritten: '{rewritten_query}'")
        
        query_result = query_info.copy()
        query_result['rewritten_query'] = rewritten_query
        rewritten_queries.append(query_result)
        print()
    
    # Step 2: Video Segmentation
    print("Step 2: Segmenting videos into events...")
    all_video_events = {}
    
    for video_id, features in video_features.items():
        metadata = video_metadata[video_id]
        print(f"  Processing {video_id} ({metadata['duration']}s)...")
        
        # Segment video (using dummy events since no VLM available)
        events = besm.segment_video(
            features, metadata['duration'], metadata['fps'], vlm_model=None
        )
        
        all_video_events[video_id] = events
        print(f"  ✓ Found {len(events)} events")
    
    # Step 3: Query-Event Matching
    print("\nStep 3: Matching queries to events...")
    predictions = []
    ground_truth = []
    
    for query_result in rewritten_queries:
        video_id = query_result['video_id']
        rewritten_query = query_result['rewritten_query']
        
        # Get events for this video
        events = all_video_events.get(video_id, [])
        
        # Match query to events
        predicted_span = qem.match_query_to_events(rewritten_query, events)
        gt_span = (query_result['start_time'], query_result['end_time'])
        
        predictions.append(predicted_span)
        ground_truth.append(gt_span)
        
        print(f"  Query: '{query_result['query'][:50]}...'")
        print(f"  Predicted: {predicted_span[0]:.1f}s - {predicted_span[1]:.1f}s")
        print(f"  Ground Truth: {gt_span[0]:.1f}s - {gt_span[1]:.1f}s")
        
        iou = compute_iou(predicted_span, gt_span)
        print(f"  IoU: {iou:.3f}")
        print()
    
    # Evaluation
    print(" Evaluation Results:")
    metrics = qem.evaluate_predictions(predictions, ground_truth)
    print_evaluation_metrics(metrics)
    
    # Summary
    print(f"\nSuccessfully processed {len(queries)} queries!")
    print(f"   Average mIoU: {metrics['mIoU']:.3f}")
    print(f"   Recall@0.3: {metrics['R@0.3']:.3f}")
    
    return {
        'queries': rewritten_queries,
        'events': all_video_events,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'metrics': metrics
    }


if __name__ == "__main__":
    try:
        results = run_simple_example()
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc() 