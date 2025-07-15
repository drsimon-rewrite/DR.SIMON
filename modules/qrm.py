"""
Query Rewriting Module (QRM)

This module rewrites medical queries into visually explicit language by:
1. Leveraging global visual context from video frames
2. Generating action-focused descriptions using a frozen LLM
3. Merging semantically similar sentences to create concise rewrites
"""

import json
import re
import torch
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from pathlib import Path


class QueryRewritingModule:
    """
    Query Rewriting Module that transforms medical terminology into 
    visually explicit language using video context.
    """
    
    def __init__(self, 
                 sentence_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.90,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Query Rewriting Module.
        
        Args:
            sentence_model: Name of the sentence transformer model
            similarity_threshold: Threshold for merging similar sentences
            device: Device to run computations on
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        # Initialize sentence embedder
        self.embedder = SentenceTransformer(sentence_model)
        self.embedder.to(device)
        
        # Patterns for text cleaning
        self.word_remove_patterns = [
            r'(?i)\bIn the video,?\s*',   # "In the video,"
            r'(?i)\bvideo\b',             # word "video"
        ]
        
        self.sentence_drop_patterns = [
            r'(?i)\boverall\b',
            r'(?i)\bhelpful\b',
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unnecessary words and phrases.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Remove specific words
        for pattern in self.word_remove_patterns:
            text = re.sub(pattern, '', text)
        
        # Split into sentences and filter
        sentences = text.split('.')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence should be dropped
            should_drop = any(re.search(pattern, sentence) 
                            for pattern in self.sentence_drop_patterns)
            
            if not should_drop:
                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences)
    
    def merge_similar_sentences(self, sentences: List[str]) -> List[str]:
        """
        Merge semantically similar sentences based on embedding similarity.
        
        Args:
            sentences: List of sentences to merge
            
        Returns:
            List of merged sentences
        """
        if len(sentences) <= 1:
            return sentences
        
        # Get embeddings
        embeddings = self.embedder.encode(sentences, convert_to_tensor=True)
        
        # Compute similarity matrix
        similarities = util.cos_sim(embeddings, embeddings)
        
        # Find groups of similar sentences
        merged = []
        used = set()
        
        for i, sentence in enumerate(sentences):
            if i in used:
                continue
                
            # Find similar sentences
            similar_indices = []
            for j in range(len(sentences)):
                if j != i and similarities[i][j] >= self.similarity_threshold:
                    similar_indices.append(j)
            
            if similar_indices:
                # Use the longest sentence as representative
                candidates = [sentence] + [sentences[j] for j in similar_indices]
                representative = max(candidates, key=len)
                merged.append(representative)
                
                # Mark as used
                used.add(i)
                used.update(similar_indices)
            else:
                merged.append(sentence)
                used.add(i)
        
        return merged
    
    def rewrite_query(self, 
                     original_query: str, 
                     video_context: str,
                     vlm_model=None) -> str:
        """
        Rewrite a medical query into visually explicit language.
        
        Args:
            original_query: The original medical query
            video_context: Visual context from video frames
            vlm_model: Vision-language model for generating rewrites
            
        Returns:
            Rewritten query in visually explicit language
        """
        # Generate action-focused description using VLM
        if vlm_model is not None:
            prompt = f"Explain '{original_query}' by describing the actions of people in the video."
            intermediate_query = vlm_model.generate(video_context, prompt)
        else:
            # Fallback: simple text processing
            intermediate_query = self._fallback_rewrite(original_query)
        
        # Clean the intermediate query
        cleaned_query = self.clean_text(intermediate_query)
        
        # Split into sentences and merge similar ones
        sentences = [s.strip() for s in cleaned_query.split('.') if s.strip()]
        merged_sentences = self.merge_similar_sentences(sentences)
        
        # Join back into final query
        final_query = '. '.join(merged_sentences[:2])  # Limit to 1-2 sentences
        
        return final_query
    
    def _fallback_rewrite(self, query: str) -> str:
        """
        Fallback rewriting method when VLM is not available.
        
        Args:
            query: Original query
            
        Returns:
            Simple rewritten query
        """
        # Simple fallback: remove medical jargon and focus on actions
        medical_terms = ['procedure', 'technique', 'method', 'process']
        action_words = ['show', 'demonstrate', 'perform', 'execute']
        
        rewritten = query.lower()
        
        # Replace medical terms with action words
        for i, term in enumerate(medical_terms):
            if term in rewritten:
                action_word = action_words[i % len(action_words)]
                rewritten = rewritten.replace(term, action_word)
        
        return rewritten
    
    def process_batch(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of queries for rewriting.
        
        Args:
            queries: List of query dictionaries
            
        Returns:
            List of processed queries with rewritten versions
        """
        processed = []
        
        for query_item in queries:
            original_query = query_item.get('query', '')
            video_context = query_item.get('video_context', '')
            
            rewritten_query = self.rewrite_query(original_query, video_context)
            
            result = query_item.copy()
            result['rewritten_query'] = rewritten_query
            result['original_query'] = original_query
            
            processed.append(result)
        
        return processed 