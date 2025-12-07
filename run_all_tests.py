#!/usr/bin/env python3
"""
Main script to run all Motivation Tests sequentially
"""

import os
import sys
import argparse
from pathlib import Path

# Add motivation_tests to path
sys.path.insert(0, str(Path(__file__).parent))

from motivation_tests.test1_token_variance import run_test1
from motivation_tests.test2_budget_violation import run_test2
from motivation_tests.test3_embedding_vs_usefulness import run_test3
from motivation_tests.test4_hard_queries import run_test4
from motivation_tests.test5_reproducibility import run_test5


def main():
    parser = argparse.ArgumentParser(description="Run ContextDB Motivation Tests")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5], 
                       help="Run specific test (1-5), or omit to run all")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Base output directory for results")
    args = parser.parse_args()
    
    tests = {
        1: ("Token Cost Variance", run_test1),
        2: ("Budget Violation → Quality Drop", run_test2),
        3: ("Embedding vs Task Usefulness", run_test3),
        4: ("Hard Queries Need Adaptive Retrieval", run_test4),
        5: ("Non-deterministic → Need Reproducibility", run_test5),
    }
    
    if args.test:
        # Run specific test
        test_num = args.test
        if test_num not in tests:
            print(f"Error: Test {test_num} not found")
            return 1
        
        test_name, test_func = tests[test_num]
        print(f"\n{'='*80}")
        print(f"Running Test {test_num}: {test_name}")
        print(f"{'='*80}\n")
        
        try:
            output_dir = os.path.join(args.output_dir, f"test{test_num}")
            test_func(output_dir=output_dir)
            print(f"\n✓ Test {test_num} completed successfully!")
            return 0
        except Exception as e:
            print(f"\n✗ Test {test_num} failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        # Run all tests sequentially
        print(f"\n{'='*80}")
        print("Running All Motivation Tests for ContextDB-0")
        print(f"{'='*80}\n")
        
        results = {}
        for test_num in sorted(tests.keys()):
            test_name, test_func = tests[test_num]
            print(f"\n{'='*80}")
            print(f"Test {test_num}/5: {test_name}")
            print(f"{'='*80}\n")
            
            try:
                output_dir = os.path.join(args.output_dir, f"test{test_num}")
                test_func(output_dir=output_dir)
                results[test_num] = "PASSED"
                print(f"\n✓ Test {test_num} completed successfully!")
            except Exception as e:
                results[test_num] = f"FAILED: {str(e)}"
                print(f"\n✗ Test {test_num} failed with error:")
                print(f"  {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        print(f"\n{'='*80}")
        print("Test Summary")
        print(f"{'='*80}")
        for test_num in sorted(tests.keys()):
            status = results[test_num]
            symbol = "✓" if status == "PASSED" else "✗"
            print(f"  {symbol} Test {test_num}: {tests[test_num][0]} - {status}")
        print(f"{'='*80}\n")
        
        # Return 0 if all passed, 1 otherwise
        return 0 if all(r == "PASSED" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

