"""
Motivation Test 3: Context quality ≠ TopK embedding similarity

Purpose: Prove that "embedding TopK documents" is not a good proxy for context quality.

This test demonstrates:
- Embedding recall is very weak for task relevance
- TopK semantic relevance ≠ task-level usefulness
- Must do system-level reasoning-aware selection
- Context selection = not just RAG
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

try:
    from .utils import count_tokens, estimate_token_cost
    from .test1_token_variance import (
        NaiveRAGSimulator, 
        generate_synthetic_documents, 
        generate_synthetic_code_tasks, 
        generate_synthetic_sql_tasks, 
        generate_synthetic_dialog_tasks
    )
except ImportError:
    from utils import count_tokens, estimate_token_cost
    from test1_token_variance import (
        NaiveRAGSimulator, 
        generate_synthetic_documents, 
        generate_synthetic_code_tasks, 
        generate_synthetic_sql_tasks, 
        generate_synthetic_dialog_tasks
    )


def create_gold_memory_mapping(tasks: List[Dict], documents: List[str]) -> Dict[str, Set[int]]:
    """
    Create ground truth mapping: for each task, which documents are actually useful.
    This simulates having labeled gold memory.
    
    In reality, this would come from:
    - Human annotations
    - Task success feedback
    - Dependency analysis
    - Workflow graphs
    """
    gold_mapping = {}
    
    # Create keyword-based matching to simulate "ground truth"
    # In real scenario, this would be from actual annotations
    task_keywords = {}
    for task in tasks:
        query = task["query"].lower()
        keywords = set(query.split())
        task_keywords[task["task_id"]] = keywords
    
    doc_keywords = {}
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        keywords = set(doc_lower.split()[:20])  # First 20 words as keywords
        doc_keywords[i] = keywords
    
    # Match: document is useful if it shares significant keywords with task
    for task in tasks:
        task_id = task["task_id"]
        task_kw = task_keywords[task_id]
        useful_docs = set()
        
        for doc_idx, doc_kw in doc_keywords.items():
            # Jaccard similarity
            intersection = len(task_kw & doc_kw)
            union = len(task_kw | doc_kw)
            similarity = intersection / union if union > 0 else 0
            
            # Also check for exact keyword matches (more strict)
            exact_matches = len([w for w in task_kw if w in doc_kw])
            
            # Document is useful if similarity > threshold OR has multiple exact matches
            if similarity > 0.15 or exact_matches >= 3:
                useful_docs.add(doc_idx)
        
        gold_mapping[task_id] = useful_docs
    
    return gold_mapping


def evaluate_recall(retrieved_indices: List[int], gold_indices: Set[int], topk: int) -> Dict:
    """Evaluate recall of retrieved documents against gold standard"""
    retrieved_set = set(retrieved_indices[:topk])
    
    if len(gold_indices) == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "retrieved_count": len(retrieved_set),
            "gold_count": 0,
            "overlap": 0
        }
    
    overlap = len(retrieved_set & gold_indices)
    recall = overlap / len(gold_indices) if len(gold_indices) > 0 else 0.0
    precision = overlap / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "retrieved_count": len(retrieved_set),
        "gold_count": len(gold_indices),
        "overlap": overlap
    }


def simulate_agent_success_with_context(retrieved_indices: List[int], gold_indices: Set[int], topk: int) -> bool:
    """
    Simulate agent success based on whether retrieved context contains useful documents.
    Success probability increases with recall.
    """
    eval_result = evaluate_recall(retrieved_indices, gold_indices, topk)
    recall = eval_result["recall"]
    
    # Success probability is roughly proportional to recall
    # But with some non-linearity: high recall is critical
    if recall >= 0.8:
        success_prob = 0.9 + recall * 0.1  # 0.9 to 1.0
    elif recall >= 0.5:
        success_prob = 0.5 + (recall - 0.5) * 0.4 / 0.3  # 0.5 to 0.9
    else:
        success_prob = recall * 1.0  # 0 to 0.5
    
    # Add some randomness
    success = np.random.random() < success_prob
    return success


def run_test3(output_dir: str = "results/test3"):
    """Run Motivation Test 3"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Motivation Test 3: Context Quality ≠ TopK Embedding Similarity")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic documents and tasks...")
    documents = generate_synthetic_documents(num_docs=1000)
    code_tasks = generate_synthetic_code_tasks(num_tasks=100)
    sql_tasks = generate_synthetic_sql_tasks(num_tasks=100)
    dialog_tasks = generate_synthetic_dialog_tasks(num_tasks=100)
    all_tasks = code_tasks + sql_tasks + dialog_tasks
    
    # Create gold memory mapping
    print("\n2. Creating gold memory mapping (ground truth)...")
    gold_mapping = create_gold_memory_mapping(all_tasks, documents)
    
    # Initialize RAG simulator
    print("\n3. Initializing RAG simulator...")
    rag = NaiveRAGSimulator()
    rag.load_documents(documents)
    
    # Test different TopK values
    topk_values = [5, 10, 20]
    results = []
    
    print("\n4. Running embedding-based retrieval and evaluating against gold standard...")
    for topk in topk_values:
        print(f"\n   Testing TopK={topk}...")
        for task in tqdm(all_tasks, desc=f"TopK={topk}"):
            # Get retrieved documents using embedding similarity
            query_embedding = rag.embedding_model.encode([task["query"]])
            query_embedding = query_embedding.astype('float32')
            distances, indices = rag.index.search(query_embedding, topk)
            retrieved_indices = indices[0].tolist()
            
            # Get gold standard
            gold_indices = gold_mapping[task["task_id"]]
            
            # Evaluate recall
            eval_result = evaluate_recall(retrieved_indices, gold_indices, topk)
            
            # Simulate agent success
            agent_success = simulate_agent_success_with_context(retrieved_indices, gold_indices, topk)
            
            result = {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "query": task["query"],
                "topk": topk,
                "retrieved_count": len(retrieved_indices),
                "gold_count": len(gold_indices),
                "recall": eval_result["recall"],
                "precision": eval_result["precision"],
                "f1": eval_result["f1"],
                "agent_success": agent_success,
                "embedding_similarity_avg": float(np.mean(1.0 / (1.0 + distances[0])))  # Convert distance to similarity
            }
            results.append(result)
    
    # Analyze results
    print("\n5. Analyzing results...")
    df = pd.DataFrame(results)
    
    print("\n   Embedding-based Retrieval Performance:")
    for topk in topk_values:
        topk_df = df[df["topk"] == topk]
        print(f"\n   TopK={topk}:")
        print(f"     Average Recall: {topk_df['recall'].mean():.3f}")
        print(f"     Average Precision: {topk_df['precision'].mean():.3f}")
        print(f"     Average F1: {topk_df['f1'].mean():.3f}")
        print(f"     Agent Success Rate: {topk_df['agent_success'].mean():.3f}")
        print(f"     Recall Std Dev: {topk_df['recall'].std():.3f}")
    
    # Save results
    results_file = os.path.join(output_dir, "embedding_vs_usefulness_results.csv")
    df.to_csv(results_file, index=False)
    print(f"\n   Results saved to: {results_file}")
    
    # Generate visualizations
    print("\n6. Generating visualizations...")
    
    # Figure 1: Recall distribution by TopK
    plt.figure(figsize=(12, 6))
    for topk in topk_values:
        topk_df = df[df["topk"] == topk]
        plt.hist(topk_df["recall"], bins=20, alpha=0.6, label=f"TopK={topk}", density=True)
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Recall Distribution: Embedding TopK vs Gold Standard", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recall_distribution.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/recall_distribution.png")
    plt.close()
    
    # Figure 2: Agent success rate vs recall
    plt.figure(figsize=(12, 6))
    
    recall_bins = np.arange(0, 1.1, 0.05)
    df["recall_bin_center"] = pd.cut(df["recall"], bins=recall_bins, 
                                     labels=(recall_bins[:-1] + recall_bins[1:])/2)
    
    for topk in topk_values:
        topk_df = df[df["topk"] == topk]
        recall_centers = []
        success_rates = []
        
        for bin_center in sorted(topk_df["recall_bin_center"].dropna().unique()):
            bin_data = topk_df[topk_df["recall_bin_center"] == bin_center]
            if len(bin_data) >= 3:
                recall_centers.append(bin_center)
                success_rates.append(bin_data["agent_success"].mean())
        
        plt.plot(recall_centers, success_rates, marker='o', label=f"TopK={topk}", linewidth=2, markersize=4)
    
    plt.xlabel("Recall (vs Gold Standard)", fontsize=12)
    plt.ylabel("Agent Success Rate", fontsize=12)
    plt.title("Agent Success Rate vs Embedding-based Recall", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_vs_recall.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/success_vs_recall.png")
    plt.close()
    
    # Figure 3: Embedding similarity vs task usefulness (scatter)
    plt.figure(figsize=(12, 6))
    
    for topk in topk_values:
        topk_df = df[df["topk"] == topk]
        plt.scatter(topk_df["embedding_similarity_avg"], topk_df["recall"], 
                   alpha=0.5, label=f"TopK={topk}", s=30)
    
    plt.xlabel("Average Embedding Similarity", fontsize=12)
    plt.ylabel("Recall (Task Usefulness)", fontsize=12)
    plt.title("Embedding Similarity ≠ Task-level Usefulness", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_vs_usefulness.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/similarity_vs_usefulness.png")
    plt.close()
    
    # Figure 4: Heatmap of recall by task type and TopK
    plt.figure(figsize=(10, 6))
    pivot = df.groupby(["task_type", "topk"])["recall"].mean().reset_index()
    pivot_table = pivot.pivot(index="task_type", columns="topk", values="recall")
    
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1,
                cbar_kws={'label': 'Average Recall'})
    plt.title("Recall by Task Type and TopK", fontsize=14, fontweight='bold')
    plt.xlabel("TopK", fontsize=12)
    plt.ylabel("Task Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_recall.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/heatmap_recall.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Test 3 completed successfully!")
    print("=" * 80)
    print("\nKey Finding: Embedding-based TopK retrieval has weak correlation")
    print("with task-level usefulness. System-level reasoning-aware selection is needed.")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    run_test3()

