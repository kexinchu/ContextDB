# ContextDB Motivation Tests

This directory contains 5 motivation tests that demonstrate the need for ContextDB.

## Setup

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Ensure you have write permissions for the results directory.

## Running Tests

### Run All Tests
```bash
cd /workspace/ContextDB
python run_all_tests.py
```

### Run Specific Test
```bash
python run_all_tests.py --test 1  # Run test 1 only
python run_all_tests.py --test 2  # Run test 2 only
# ... etc
```

### Run Individual Tests Directly
```bash
cd motivation_tests
python test1_token_variance.py
python test2_budget_violation.py
python test3_embedding_vs_usefulness.py
python test4_hard_queries.py
python test5_reproducibility.py
```

## Test Descriptions

### Test 1: Token Cost Variance (10-100×)
**Purpose**: Prove that heuristic-based RAG token costs are extremely unstable.

**Outputs**:
- `results/test1/token_cost_results.csv` - Raw results
- `results/test1/cdf_token_usage.png` - CDF of token usage
- `results/test1/heatmap_task_variance.png` - Variance heatmap
- `results/test1/boxplot_token_costs.png` - Cost distribution

### Test 2: Budget Violation → Quality Drop
**Purpose**: Prove that token budget affects agent success rate.

**Outputs**:
- `results/test2/budget_violation_results.csv` - Raw results
- `results/test2/success_rate_vs_deviation.png` - Success vs deviation
- `results/test2/success_rate_vs_utilization.png` - Success vs utilization
- `results/test2/heatmap_success_rate.png` - Success rate heatmap

### Test 3: Embedding vs Task Usefulness
**Purpose**: Prove that embedding TopK ≠ task-level usefulness.

**Outputs**:
- `results/test3/embedding_vs_usefulness_results.csv` - Raw results
- `results/test3/recall_distribution.png` - Recall distribution
- `results/test3/success_vs_recall.png` - Success vs recall
- `results/test3/similarity_vs_usefulness.png` - Similarity scatter
- `results/test3/heatmap_recall.png` - Recall heatmap

### Test 4: Hard Queries Need Adaptive Retrieval
**Purpose**: Demonstrate that easy/hard queries need different strategies.

**Outputs**:
- `results/test4/hard_queries_results.csv` - Raw results
- `results/test4/cost_by_hardness.png` - Cost comparison
- `results/test4/success_by_hardness.png` - Success comparison
- `results/test4/cost_vs_success.png` - Cost-success trade-off
- `results/test4/hardness_distribution.png` - Hardness distribution

### Test 5: Non-deterministic → Need Reproducibility
**Purpose**: Show that naive RAG is non-deterministic and needs DB-level solutions.

**Outputs**:
- `results/test5/reproducibility_results.csv` - Raw results
- `results/test5/variation_across_runs.png` - Variation metrics
- `results/test5/cost_variation.png` - Cost variation
- `results/test5/stability_distribution.png` - Stability distribution
- `results/test5/cost_cv_by_type.png` - Cost CV by type
- `results/test5/stability_correlation.png` - Stability correlation

## Notes

- Tests use synthetic data for reproducibility
- Embedding model: `all-MiniLM-L6-v2` (downloaded automatically)
- Tests may take several minutes to run (especially embedding generation)
- All results are saved to CSV for further analysis

