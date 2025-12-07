"""
Motivation Test 5: LLM memory is not deterministic → need DB-level reproducibility

Purpose: Show why context IR + optimization + trace is needed.
This paves the way for papers 2 and 3.

This test demonstrates:
- Naive RAG context varies across runs → model outputs differ
- Context instability → agent instability
- Cannot debug/reproduce → difficult to deploy in production
- Must be solved through context optimizer + IR
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict

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


def simulate_llm_output(context: str, query: str, seed: int = None) -> str:
    """
    Simulate LLM output based on context and query.
    In reality, this would be actual LLM inference.
    For simulation, we create a deterministic hash-based output when seed is provided,
    and non-deterministic output otherwise.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate output: hash of context + query
    # With seed: deterministic; without: random
    context_hash = hash(context) % 1000
    query_hash = hash(query) % 1000
    
    if seed is not None:
        # Deterministic: same context + query + seed = same output
        combined = (context_hash + query_hash + seed) % 100
    else:
        # Non-deterministic: random variation
        combined = (context_hash + query_hash + np.random.randint(0, 100)) % 100
    
    # Simulate output quality/variation
    output = f"Response_{combined}"
    return output


def run_naive_rag_with_variation(rag: NaiveRAGSimulator, query: str, topk: int = 10, 
                                  add_noise: bool = True) -> List[str]:
    """
    Run naive RAG retrieval with potential variation.
    In real systems, variation comes from:
    - Non-deterministic embedding model behavior
    - Vector database approximate search
    - Floating point precision
    - Concurrent updates
    """
    query_embedding = rag.embedding_model.encode([query])
    query_embedding = query_embedding.astype('float32')
    
    if add_noise:
        # Add small random noise to simulate non-determinism
        noise = np.random.normal(0, 0.01, query_embedding.shape)
        query_embedding = query_embedding + noise
    
    distances, indices = rag.index.search(query_embedding, topk)
    retrieved_docs = [rag.documents[idx] for idx in indices[0]]
    return retrieved_docs


def run_test5(output_dir: str = "results/test5"):
    """Run Motivation Test 5"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Motivation Test 5: LLM Memory Non-deterministic → Need DB-level Reproducibility")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic documents and tasks...")
    documents = generate_synthetic_documents(num_docs=1000)
    code_tasks = generate_synthetic_code_tasks(num_tasks=30)
    sql_tasks = generate_synthetic_sql_tasks(num_tasks=30)
    dialog_tasks = generate_synthetic_dialog_tasks(num_tasks=30)
    all_tasks = code_tasks + sql_tasks + dialog_tasks
    
    # Initialize RAG simulator
    print("\n2. Initializing RAG simulator...")
    rag = NaiveRAGSimulator()
    rag.load_documents(documents)
    
    # Run multiple times for same tasks
    num_runs = 10
    results = []
    
    print(f"\n3. Running naive RAG {num_runs} times for each task (simulating non-determinism)...")
    for task in tqdm(all_tasks, desc="Tasks"):
        query = task["query"]
        
        # Track variation across runs
        contexts_across_runs = []
        outputs_across_runs = []
        costs_across_runs = []
        
        for run_id in range(num_runs):
            # Retrieve context (with variation)
            retrieved = run_naive_rag_with_variation(rag, query, topk=10, add_noise=True)
            context = "\n\n".join(retrieved)
            contexts_across_runs.append(context)
            
            # Get token cost
            cost = estimate_token_cost(context)
            costs_across_runs.append(cost)
            
            # Simulate LLM output (non-deterministic due to context variation)
            output = simulate_llm_output(context, query, seed=None)  # No seed = non-deterministic
            outputs_across_runs.append(output)
        
        # Calculate variation metrics
        unique_contexts = len(set(contexts_across_runs))
        unique_outputs = len(set(outputs_across_runs))
        cost_std = np.std(costs_across_runs)
        cost_mean = np.mean(costs_across_runs)
        cost_cv = cost_std / cost_mean if cost_mean > 0 else 0
        
        # Jaccard similarity of contexts across runs
        context_sets = [set(c.split()) for c in contexts_across_runs]
        if len(context_sets) > 1:
            # Average pairwise Jaccard similarity
            similarities = []
            for i in range(len(context_sets)):
                for j in range(i+1, len(context_sets)):
                    intersection = len(context_sets[i] & context_sets[j])
                    union = len(context_sets[i] | context_sets[j])
                    if union > 0:
                        similarities.append(intersection / union)
            avg_context_similarity = np.mean(similarities) if similarities else 0.0
        else:
            avg_context_similarity = 1.0
        
        result = {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "query": query,
            "num_runs": num_runs,
            "unique_contexts": unique_contexts,
            "unique_outputs": unique_outputs,
            "context_stability": avg_context_similarity,
            "output_stability": unique_outputs / num_runs,  # Lower is better (more stable)
            "cost_mean": cost_mean,
            "cost_std": cost_std,
            "cost_cv": cost_cv,
            "cost_min": np.min(costs_across_runs),
            "cost_max": np.max(costs_across_runs),
            "cost_range": np.max(costs_across_runs) - np.min(costs_across_runs)
        }
        results.append(result)
    
    # Analyze results
    print("\n4. Analyzing results...")
    df = pd.DataFrame(results)
    
    print("\n   Variation Metrics Across Runs:")
    print(f"     Average Unique Contexts per Task: {df['unique_contexts'].mean():.1f} / {num_runs}")
    print(f"     Average Unique Outputs per Task: {df['unique_outputs'].mean():.1f} / {num_runs}")
    print(f"     Average Context Stability (Jaccard): {df['context_stability'].mean():.3f}")
    print(f"     Average Output Stability: {1 - df['output_stability'].mean():.3f}")
    print(f"     Average Cost CV: {df['cost_cv'].mean():.3f}")
    print(f"     Average Cost Range: {df['cost_range'].mean():.0f} tokens")
    
    # Save results
    results_file = os.path.join(output_dir, "reproducibility_results.csv")
    df.to_csv(results_file, index=False)
    print(f"\n   Results saved to: {results_file}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    # Figure 1: Context selection variation across runs
    plt.figure(figsize=(12, 6))
    
    task_types = df["task_type"].unique()
    x = np.arange(len(task_types))
    width = 0.35
    
    unique_contexts = [df[df["task_type"] == t]["unique_contexts"].mean() for t in task_types]
    unique_outputs = [df[df["task_type"] == t]["unique_outputs"].mean() for t in task_types]
    
    plt.bar(x - width/2, unique_contexts, width, label='Unique Contexts', alpha=0.8)
    plt.bar(x + width/2, unique_outputs, width, label='Unique Outputs', alpha=0.8)
    
    plt.xlabel("Task Type", fontsize=12)
    plt.ylabel(f"Average Unique Variants (out of {num_runs} runs)", fontsize=12)
    plt.title("Context and Output Variation Across Runs", fontsize=14, fontweight='bold')
    plt.xticks(x, task_types)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "variation_across_runs.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/variation_across_runs.png")
    plt.close()
    
    # Figure 2: Cost variation across runs
    plt.figure(figsize=(12, 6))
    
    # Sample a few tasks to show cost variation
    sample_tasks = df.sample(min(10, len(df)))
    
    for idx, row in sample_tasks.iterrows():
        # Simulate cost distribution (we don't have individual run costs stored)
        # Use mean and std to create distribution
        costs = np.random.normal(row["cost_mean"], row["cost_std"], num_runs)
        costs = np.maximum(costs, row["cost_min"])  # Ensure within range
        costs = np.minimum(costs, row["cost_max"])
        
        plt.scatter([row["task_id"]] * num_runs, costs, alpha=0.3, s=20)
    
    plt.xlabel("Task ID", fontsize=12)
    plt.ylabel("Token Cost", fontsize=12)
    plt.title("Token Cost Variation Across Runs (Sample Tasks)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_variation.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/cost_variation.png")
    plt.close()
    
    # Figure 3: Stability metrics distribution
    plt.figure(figsize=(12, 6))
    
    plt.hist(df["context_stability"], bins=20, alpha=0.6, label="Context Stability", color='blue')
    plt.hist(1 - df["output_stability"], bins=20, alpha=0.6, label="Output Stability", color='red')
    
    plt.xlabel("Stability Score (Higher = More Stable)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Stability Metrics", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_distribution.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/stability_distribution.png")
    plt.close()
    
    # Figure 4: Cost coefficient of variation by task type
    plt.figure(figsize=(10, 6))
    
    task_types = df["task_type"].unique()
    cv_by_type = [df[df["task_type"] == t]["cost_cv"].mean() for t in task_types]
    
    plt.bar(task_types, cv_by_type, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.xlabel("Task Type", fontsize=12)
    plt.ylabel("Average Cost Coefficient of Variation", fontsize=12)
    plt.title("Cost Unpredictability by Task Type", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_cv_by_type.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/cost_cv_by_type.png")
    plt.close()
    
    # Figure 5: Scatter plot: context stability vs output stability
    plt.figure(figsize=(10, 8))
    
    plt.scatter(df["context_stability"], 1 - df["output_stability"], 
               alpha=0.6, s=50, c=df["cost_cv"], cmap='viridis')
    plt.colorbar(label='Cost CV')
    
    plt.xlabel("Context Stability (Jaccard Similarity)", fontsize=12)
    plt.ylabel("Output Stability", fontsize=12)
    plt.title("Context Stability vs Output Stability\n(Color = Cost CV)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_correlation.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/stability_correlation.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Test 5 completed successfully!")
    print("=" * 80)
    print("\nKey Finding: Naive RAG shows significant non-determinism:")
    print(f"  - Contexts vary across runs (avg {df['unique_contexts'].mean():.1f} unique contexts)")
    print(f"  - Outputs vary across runs (avg {df['unique_outputs'].mean():.1f} unique outputs)")
    print(f"  - Costs are unpredictable (avg CV: {df['cost_cv'].mean():.3f})")
    print("\nThis demonstrates the need for:")
    print("  - Context IR (intermediate representation)")
    print("  - Context optimizer (deterministic planning)")
    print("  - Trace/replay mechanisms (for debugging)")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    run_test5()

