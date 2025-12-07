"""
Motivation Test 2: Token budget violation → drastic quality drop

Purpose: Prove that token budget affects agent success rate, and naive heuristics 
easily exceed or fall below budget.

This test demonstrates:
- Success rate vs budget deviation (non-linear drop)
- Naive RAG doesn't know token limits, randomly exceeds/shrinks
- Agent outputs degrade vs success is critical
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

try:
    from .utils import count_tokens, estimate_token_cost, truncate_to_budget
    from .test1_token_variance import (
        NaiveRAGSimulator, 
        generate_synthetic_documents, 
        generate_synthetic_code_tasks, 
        generate_synthetic_sql_tasks, 
        generate_synthetic_dialog_tasks
    )
except ImportError:
    from utils import count_tokens, estimate_token_cost, truncate_to_budget
    from test1_token_variance import (
        NaiveRAGSimulator, 
        generate_synthetic_documents, 
        generate_synthetic_code_tasks, 
        generate_synthetic_sql_tasks, 
        generate_synthetic_dialog_tasks
    )


def simulate_agent_task(context: str, query: str, budget: int) -> Dict:
    """
    Simulate agent task execution with context.
    Returns success status based on whether context is sufficient and within budget.
    """
    # Check if context fits budget
    context_tokens = estimate_token_cost(context)
    
    # Truncate if exceeds budget
    if context_tokens > budget:
        context = truncate_to_budget(context, budget)
        context_tokens = estimate_token_cost(context)
        # Exceeding budget leads to quality degradation
        # Simulate: if we had to truncate significantly, success rate drops
        truncation_ratio = context_tokens / budget if budget > 0 else 0
        # Success probability decreases as truncation increases
        success_prob = max(0, 1.0 - (1.0 - truncation_ratio) * 2.0)
    else:
        # If context is too small, also leads to failure
        utilization_ratio = context_tokens / budget if budget > 0 else 0
        # Too little context (less than 50% of budget) leads to failure
        if utilization_ratio < 0.5:
            success_prob = utilization_ratio * 1.5  # Linear increase from 0 to 0.75
        else:
            # Optimal range: 70-100% utilization
            if 0.7 <= utilization_ratio <= 1.0:
                success_prob = 0.9 + (utilization_ratio - 0.7) * 0.1 / 0.3  # 0.9 to 1.0
            else:
                success_prob = 0.7 + (utilization_ratio - 0.5) * 0.2 / 0.2  # 0.7 to 0.9
    
    # Add some randomness
    success = np.random.random() < success_prob
    
    return {
        "success": success,
        "context_tokens": context_tokens,
        "budget": budget,
        "deviation": context_tokens - budget,
        "deviation_ratio": (context_tokens - budget) / budget if budget > 0 else 0,
        "utilization": context_tokens / budget if budget > 0 else 0,
        "success_prob": success_prob
    }


def run_test2(output_dir: str = "results/test2"):
    """Run Motivation Test 2"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Motivation Test 2: Token Budget Violation → Quality Drop")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic documents and tasks...")
    documents = generate_synthetic_documents(num_docs=1000)
    code_tasks = generate_synthetic_code_tasks(num_tasks=50)
    sql_tasks = generate_synthetic_sql_tasks(num_tasks=50)
    dialog_tasks = generate_synthetic_dialog_tasks(num_tasks=50)
    all_tasks = code_tasks + sql_tasks + dialog_tasks
    
    # Initialize RAG simulator
    print("\n2. Initializing RAG simulator...")
    rag = NaiveRAGSimulator()
    rag.load_documents(documents)
    
    # Test different token budgets
    budgets = [4000, 8000, 16000]
    topk_values = [5, 10, 20]
    results = []
    
    print("\n3. Running naive RAG with different budgets and TopK values...")
    for budget in budgets:
        for topk in topk_values:
            print(f"\n   Testing Budget={budget}, TopK={topk}...")
            for task in tqdm(all_tasks, desc=f"Budget={budget}, TopK={topk}"):
                # Get context using naive RAG
                retrieved = rag.retrieve(task["query"], topk=topk)
                context = "\n\n".join(retrieved)
                
                # Simulate agent task
                result = simulate_agent_task(context, task["query"], budget)
                result.update({
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "query": task["query"],
                    "budget": budget,
                    "topk": topk
                })
                results.append(result)
    
    # Analyze results
    print("\n4. Analyzing results...")
    df = pd.DataFrame(results)
    
    # Calculate success rate by budget deviation
    print("\n   Success Rate by Budget Deviation:")
    df["deviation_bin"] = pd.cut(df["deviation_ratio"], bins=np.arange(-1.0, 1.5, 0.1))
    success_by_deviation = df.groupby("deviation_bin")["success"].agg(["mean", "count"])
    print(success_by_deviation)
    
    # Save results
    results_file = os.path.join(output_dir, "budget_violation_results.csv")
    df.to_csv(results_file, index=False)
    print(f"\n   Results saved to: {results_file}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    # Figure 1: Success rate vs budget deviation
    plt.figure(figsize=(12, 6))
    
    # Create deviation bins
    bins = np.arange(-1.0, 1.5, 0.05)
    df["deviation_bin_center"] = pd.cut(df["deviation_ratio"], bins=bins, labels=(bins[:-1] + bins[1:])/2)
    
    for budget in budgets:
        budget_df = df[df["budget"] == budget]
        deviation_centers = []
        success_rates = []
        counts = []
        
        for bin_center in sorted(budget_df["deviation_bin_center"].dropna().unique()):
            bin_data = budget_df[budget_df["deviation_bin_center"] == bin_center]
            if len(bin_data) > 0:
                deviation_centers.append(bin_center)
                success_rates.append(bin_data["success"].mean())
                counts.append(len(bin_data))
        
        # Filter out bins with too few samples
        deviation_centers = [d for d, c in zip(deviation_centers, counts) if c >= 3]
        success_rates = [s for s, c in zip(success_rates, counts) if c >= 3]
        
        plt.plot(deviation_centers, success_rates, marker='o', label=f"Budget={budget}", linewidth=2, markersize=4)
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Exact Budget')
    plt.xlabel("Budget Deviation Ratio (Actual - Budget) / Budget", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.title("Success Rate vs Budget Deviation (Non-linear Drop)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate_vs_deviation.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/success_rate_vs_deviation.png")
    plt.close()
    
    # Figure 2: Success rate by utilization ratio
    plt.figure(figsize=(12, 6))
    
    utilization_bins = np.arange(0, 2.0, 0.05)
    df["utilization_bin_center"] = pd.cut(df["utilization"], bins=utilization_bins, 
                                          labels=(utilization_bins[:-1] + utilization_bins[1:])/2)
    
    for budget in budgets:
        budget_df = df[df["budget"] == budget]
        utilization_centers = []
        success_rates = []
        
        for bin_center in sorted(budget_df["utilization_bin_center"].dropna().unique()):
            bin_data = budget_df[budget_df["utilization_bin_center"] == bin_center]
            if len(bin_data) >= 3:
                utilization_centers.append(bin_center)
                success_rates.append(bin_data["success"].mean())
        
        plt.plot(utilization_centers, success_rates, marker='o', label=f"Budget={budget}", linewidth=2, markersize=4)
    
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='100% Utilization')
    plt.xlabel("Budget Utilization Ratio", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.title("Success Rate vs Budget Utilization", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate_vs_utilization.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/success_rate_vs_utilization.png")
    plt.close()
    
    # Figure 3: Heatmap of success rate by budget and TopK
    plt.figure(figsize=(10, 6))
    pivot = df.groupby(["budget", "topk"])["success"].mean().reset_index()
    pivot_table = pivot.pivot(index="budget", columns="topk", values="success")
    
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={'label': 'Success Rate'})
    plt.title("Success Rate by Budget and TopK", fontsize=14, fontweight='bold')
    plt.xlabel("TopK", fontsize=12)
    plt.ylabel("Budget (tokens)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_success_rate.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/heatmap_success_rate.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Test 2 completed successfully!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    run_test2()

