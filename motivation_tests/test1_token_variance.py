"""
Motivation Test 1: Naive RAG token cost variance is 10-100×

Purpose: Prove that heuristic-based RAG token costs are extremely unstable and unpredictable.

This test demonstrates:
- Token cost distribution: P50 = X, P95 = 10X, P99 = 50X
- Naive selection strategies lead to extreme token overhead
- Upstream LLM budget and downstream latency are unpredictable
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
    from .utils import count_tokens, estimate_token_cost, calculate_percentiles
except ImportError:
    from utils import count_tokens, estimate_token_cost, calculate_percentiles


class NaiveRAGSimulator:
    """Simulate naive RAG with different TopK values"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model"""
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def load_documents(self, documents: List[str]):
        """Load documents and create embeddings"""
        self.documents = documents
        print(f"Creating embeddings for {len(documents)} documents...")
        self.embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        self.embeddings = self.embeddings.astype('float32')
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, topk: int = 5) -> List[str]:
        """Retrieve top-k documents using naive embedding similarity"""
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        
        distances, indices = self.index.search(query_embedding, topk)
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        return retrieved_docs
    
    def get_context_token_cost(self, query: str, topk: int = 5) -> int:
        """Get token cost for retrieved context"""
        retrieved = self.retrieve(query, topk)
        context = "\n\n".join(retrieved)
        return estimate_token_cost(context)


def generate_synthetic_code_tasks(num_tasks: int = 100) -> List[Dict]:
    """Generate synthetic code tasks"""
    tasks = []
    base_queries = [
        "How to implement a binary search tree?",
        "What is the difference between list and tuple in Python?",
        "How to handle exceptions in Java?",
        "Explain the observer pattern with example",
        "How to optimize database queries?",
        "What is async/await in JavaScript?",
        "How to implement a REST API?",
        "Explain dependency injection",
        "How to use decorators in Python?",
        "What is the difference between stack and queue?",
    ]
    
    for i in range(num_tasks):
        query = np.random.choice(base_queries)
        tasks.append({
            "task_id": f"code_{i}",
            "query": query,
            "task_type": "code"
        })
    return tasks


def generate_synthetic_sql_tasks(num_tasks: int = 100) -> List[Dict]:
    """Generate synthetic SQL tasks"""
    tasks = []
    base_queries = [
        "How to join multiple tables in SQL?",
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "How to optimize a slow SQL query?",
        "Explain window functions in SQL",
        "How to handle NULL values in SQL?",
        "What is a subquery and when to use it?",
        "How to create an index for better performance?",
        "Explain ACID properties in databases",
        "How to write a recursive CTE?",
        "What is the difference between DELETE and TRUNCATE?",
    ]
    
    for i in range(num_tasks):
        query = np.random.choice(base_queries)
        tasks.append({
            "task_id": f"sql_{i}",
            "query": query,
            "task_type": "sql"
        })
    return tasks


def generate_synthetic_dialog_tasks(num_tasks: int = 100) -> List[Dict]:
    """Generate synthetic multi-turn dialog tasks"""
    tasks = []
    base_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does gradient descent work?",
        "What is overfitting and how to prevent it?",
        "Explain the difference between supervised and unsupervised learning",
        "How to evaluate a machine learning model?",
        "What is cross-validation?",
        "Explain regularization techniques",
        "How to handle imbalanced datasets?",
        "What is feature engineering?",
    ]
    
    for i in range(num_tasks):
        query = np.random.choice(base_queries)
        tasks.append({
            "task_id": f"dialog_{i}",
            "query": query,
            "task_type": "dialog"
        })
    return tasks


def generate_synthetic_documents(num_docs: int = 1000, doc_length_range: Tuple[int, int] = (100, 2000)) -> List[str]:
    """Generate synthetic documents with varying lengths"""
    documents = []
    
    # Code-related documents
    code_topics = [
        "Binary search tree implementation with detailed examples",
        "Python list vs tuple: mutability, performance, use cases",
        "Exception handling in Java: try-catch, finally, custom exceptions",
        "Observer pattern: definition, UML diagram, code examples",
        "Database query optimization: indexing, query plans, caching",
        "JavaScript async/await: promises, error handling, best practices",
        "REST API design: endpoints, HTTP methods, status codes",
        "Dependency injection: IoC containers, benefits, examples",
        "Python decorators: syntax, use cases, advanced patterns",
        "Stack vs Queue: data structures, operations, applications",
    ]
    
    # SQL-related documents
    sql_topics = [
        "SQL JOIN operations: INNER, LEFT, RIGHT, FULL OUTER JOIN",
        "SQL query optimization: indexes, execution plans, statistics",
        "Window functions: ROW_NUMBER, RANK, DENSE_RANK, aggregate functions",
        "NULL handling: IS NULL, COALESCE, NULLIF, three-valued logic",
        "Subqueries: correlated, uncorrelated, EXISTS, IN, scalar subqueries",
        "Database indexing: B-tree, hash indexes, composite indexes",
        "ACID properties: atomicity, consistency, isolation, durability",
        "Recursive CTEs: hierarchical queries, graph traversal",
        "SQL DELETE vs TRUNCATE: differences, use cases, performance",
        "SQL best practices: normalization, denormalization, query patterns",
    ]
    
    # Dialog/ML-related documents
    ml_topics = [
        "Machine learning introduction: supervised, unsupervised, reinforcement",
        "Neural networks: perceptrons, backpropagation, activation functions",
        "Gradient descent: batch, stochastic, mini-batch, learning rate",
        "Overfitting: causes, detection, prevention techniques",
        "Model evaluation: accuracy, precision, recall, F1, ROC curve",
        "Cross-validation: k-fold, stratified, leave-one-out",
        "Regularization: L1, L2, dropout, early stopping",
        "Imbalanced datasets: SMOTE, undersampling, class weights",
        "Feature engineering: scaling, encoding, selection, extraction",
        "Deep learning: CNNs, RNNs, transformers, attention mechanisms",
    ]
    
    all_topics = code_topics + sql_topics + ml_topics
    
    for i in range(num_docs):
        topic = np.random.choice(all_topics)
        # Vary document length significantly
        length = np.random.randint(doc_length_range[0], doc_length_range[1])
        # Create document by repeating and expanding the topic
        doc = topic + " " * 10  # Base
        # Expand to desired length
        words = topic.split()
        while len(doc) < length:
            doc += " " + " ".join(np.random.choice(words, size=min(10, len(words)), replace=True))
        documents.append(doc[:length])
    
    return documents


def run_test1(output_dir: str = "results/test1"):
    """Run Motivation Test 1"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Motivation Test 1: Naive RAG Token Cost Variance")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic documents and tasks...")
    documents = generate_synthetic_documents(num_docs=1000)
    code_tasks = generate_synthetic_code_tasks(num_tasks=100)
    sql_tasks = generate_synthetic_sql_tasks(num_tasks=100)
    dialog_tasks = generate_synthetic_dialog_tasks(num_tasks=100)
    all_tasks = code_tasks + sql_tasks + dialog_tasks
    
    # Initialize RAG simulator
    print("\n2. Initializing RAG simulator...")
    rag = NaiveRAGSimulator()
    rag.load_documents(documents)
    
    # Test different TopK values
    topk_values = [5, 10, 20]
    results = []
    
    print("\n3. Running naive RAG with different TopK values...")
    for topk in topk_values:
        print(f"\n   Testing TopK={topk}...")
        for task in tqdm(all_tasks, desc=f"TopK={topk}"):
            token_cost = rag.get_context_token_cost(task["query"], topk=topk)
            results.append({
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "query": task["query"],
                "topk": topk,
                "token_cost": token_cost
            })
    
    # Analyze results
    print("\n4. Analyzing results...")
    df = pd.DataFrame(results)
    
    # Calculate percentiles for each TopK
    print("\n   Token Cost Percentiles by TopK:")
    for topk in topk_values:
        costs = df[df["topk"] == topk]["token_cost"].values
        percentiles = calculate_percentiles(costs.tolist(), [50, 95, 99])
        p50 = percentiles[50]
        print(f"\n   TopK={topk}:")
        print(f"     P50: {p50:.0f} tokens")
        print(f"     P95: {percentiles[95]:.0f} tokens ({percentiles[95]/p50:.1f}x)")
        print(f"     P99: {percentiles[99]:.0f} tokens ({percentiles[99]/p50:.1f}x)")
    
    # Save results
    results_file = os.path.join(output_dir, "token_cost_results.csv")
    df.to_csv(results_file, index=False)
    print(f"\n   Results saved to: {results_file}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    # Figure 1: CDF of token usage
    plt.figure(figsize=(12, 6))
    for topk in topk_values:
        costs = df[df["topk"] == topk]["token_cost"].values
        sorted_costs = np.sort(costs)
        p = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs) * 100
        plt.plot(sorted_costs, p, label=f"TopK={topk}", linewidth=2)
    
    plt.xlabel("Token Cost", fontsize=12)
    plt.ylabel("Cumulative Percentage (%)", fontsize=12)
    plt.title("CDF of Token Usage in Naive RAG", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdf_token_usage.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/cdf_token_usage.png")
    plt.close()
    
    # Figure 2: Heatmap of task types vs token cost variance
    plt.figure(figsize=(10, 6))
    variance_data = []
    for task_type in ["code", "sql", "dialog"]:
        for topk in topk_values:
            costs = df[(df["task_type"] == task_type) & (df["topk"] == topk)]["token_cost"].values
            variance = np.var(costs)
            mean_cost = np.mean(costs)
            cv = np.std(costs) / mean_cost if mean_cost > 0 else 0  # Coefficient of variation
            variance_data.append({
                "task_type": task_type,
                "topk": topk,
                "variance": variance,
                "cv": cv,
                "mean": mean_cost
            })
    
    variance_df = pd.DataFrame(variance_data)
    pivot_cv = variance_df.pivot(index="task_type", columns="topk", values="cv")
    
    sns.heatmap(pivot_cv, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Coefficient of Variation'})
    plt.title("Token Cost Variance (CV) by Task Type and TopK", fontsize=14, fontweight='bold')
    plt.xlabel("TopK", fontsize=12)
    plt.ylabel("Task Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_task_variance.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/heatmap_task_variance.png")
    plt.close()
    
    # Figure 3: Box plot of token costs
    plt.figure(figsize=(12, 6))
    df_melted = df.melt(id_vars=["topk", "task_type"], value_vars=["token_cost"], 
                        var_name="metric", value_name="value")
    sns.boxplot(data=df, x="topk", y="token_cost", hue="task_type")
    plt.xlabel("TopK", fontsize=12)
    plt.ylabel("Token Cost", fontsize=12)
    plt.title("Token Cost Distribution by TopK and Task Type", fontsize=14, fontweight='bold')
    plt.legend(title="Task Type")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_token_costs.png"), dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/boxplot_token_costs.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("Test 1 completed successfully!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    run_test1()

