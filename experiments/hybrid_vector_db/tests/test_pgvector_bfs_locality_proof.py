import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pgvector_design1_design2_design3_selectivity_benchmark as benchmark  # noqa: E402


def locality(blocks: list[int]) -> dict[str, object]:
    pairs = max(len(blocks) - 1, 0)
    same = sum(left == right for left, right in zip(blocks, blocks[1:]))
    next_page = sum(right == left + 1 for left, right in zip(blocks, blocks[1:]))
    nondecreasing = sum(right >= left for left, right in zip(blocks, blocks[1:]))
    return {
        "format": "sqlens-hnsw-bfs-locality-v1",
        "rank_base": 0,
        "graph_nodes": len(blocks),
        "reachable_nodes": len(blocks),
        "fallback_nodes": 0,
        "sequence_nodes": len(blocks),
        "adjacent_pairs": pairs,
        "same_block_pairs": same,
        "next_block_pairs": next_page,
        "same_or_next_page_pairs": same + next_page,
        "nondecreasing_pairs": nondecreasing,
        "backward_pairs": pairs - nondecreasing,
        "total_abs_block_delta": sum(
            abs(right - left) for left, right in zip(blocks, blocks[1:])
        ),
        "max_abs_block_delta": max(
            [abs(right - left) for left, right in zip(blocks, blocks[1:])] or [0]
        ),
        "page_runs": len(blocks) and 1 + sum(
            left != right for left, right in zip(blocks, blocks[1:])
        ),
        "same_block_ratio": same / pairs if pairs else 0.0,
        "same_or_next_page_ratio": (same + next_page) / pairs if pairs else 0.0,
        "nondecreasing_ratio": nondecreasing / pairs if pairs else 0.0,
        "full_statistics": True,
        "sample_limit": 256,
        "sample_count": len(blocks),
        "sample_truncated": False,
        "sample_strategy": "evenly_spaced_inclusive",
        "rank_samples": [
            {"rank": rank, "block": block, "offset": 1}
            for rank, block in enumerate(blocks)
        ],
    }


def edge_span(graph_nodes: int, index_blocks: int = 20) -> dict[str, object]:
    def stats(directed: int) -> dict[str, object]:
        same = 2
        within_one = 4
        within_four = min(8, directed)
        within_sixteen = directed
        return {
            "directed_edges": directed,
            "same_page_edges": same,
            "within_1_page_edges": within_one,
            "within_4_pages_edges": within_four,
            "within_16_pages_edges": within_sixteen,
            "same_page_ratio": same / directed,
            "within_1_page_ratio": within_one / directed,
            "within_4_pages_ratio": within_four / directed,
            "within_16_pages_ratio": within_sixteen / directed,
            "mean_abs_block_delta": 2.5,
            "p50_abs_block_delta": 1,
            "p95_abs_block_delta": 3,
            "p99_abs_block_delta": 5,
            "max_abs_block_delta": 8,
        }

    return {
        "format": "sqlens-hnsw-edge-span-v1",
        "graph_nodes": graph_nodes,
        "index_blocks": index_blocks,
        "source_page_scope": "owner_neighbor_tuple_page",
        "target_page_scope": "destination_element_page",
        "edge_scope": "complete_directed_adjacency_with_level_duplicates",
        "full_statistics": True,
        "all_layers": stats(10),
        "level_zero": stats(8),
    }


class PgvectorBfsLocalityProofTests(unittest.TestCase):
    def test_complete_counters_and_rank_samples_are_validated(self) -> None:
        value = locality([10, 10, 11, 15, 14])
        benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

        value["sample_truncated"] = True
        with self.assertRaisesRegex(RuntimeError, "sample truncation"):
            benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

    def test_counter_ratios_page_runs_and_sample_positions_fail_closed(self) -> None:
        value = locality([10, 10, 11, 15, 14])
        value["page_runs"] = 2
        with self.assertRaisesRegex(RuntimeError, "page-run"):
            benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

        value = locality([10, 10, 11, 15, 14])
        value["same_block_ratio"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "same_block_ratio"):
            benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

        value = locality(list(range(300)))
        value["sample_limit"] = 256
        value["sample_count"] = 256
        value["sample_truncated"] = True
        value["rank_samples"] = [
            {
                "rank": index * 299 // 255,
                "block": index * 299 // 255,
                "offset": 1,
            }
            for index in range(256)
        ]
        benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")
        value["rank_samples"][100]["rank"] += 1
        with self.assertRaisesRegex(RuntimeError, "rank samples"):
            benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

    def test_rank_sample_block_and_offset_are_real_integers(self) -> None:
        for field, replacement in (("block", True), ("offset", 0)):
            value = locality([10, 11])
            value["rank_samples"][0][field] = replacement
            with self.assertRaisesRegex(RuntimeError, "rank samples"):
                benchmark.validate_d2_bfs_locality(value, "left_bfs_locality")

    def test_stable_d2_proof_keeps_both_symmetric_locality_objects(self) -> None:
        comparison = {
            field: True
            for field in (
                "same_heap",
                "logical_equal",
                "physical_equal",
                "entry_equal",
                "definition_equal",
                "tuple_coverage_equal",
            )
        }
        comparison.update(
            {
                "format": "sqlens-hnsw-compare-v2",
                "left_definition_digest": "sha256:" + "1" * 64,
                "right_definition_digest": "sha256:" + "1" * 64,
                "left_tuple_coverage_digest": "sha256:" + "2" * 64,
                "right_tuple_coverage_digest": "sha256:" + "2" * 64,
                "left_logical_digest": "sha256:" + "3" * 64,
                "right_logical_digest": "sha256:" + "3" * 64,
                "left_physical_digest": "sha256:" + "4" * 64,
                "right_physical_digest": "sha256:" + "5" * 64,
                "left_bfs_locality": locality([1, 2, 2]),
                "right_bfs_locality": locality([8, 8, 9]),
            }
        )
        stable = benchmark.stable_d2_graph_proof(
            {
                "source_index": "source",
                "clone_index": "clone",
                "relations": {
                    "source": {
                        "name": "source",
                        "oid": 1,
                        "relfilenode": 2,
                        "heap_oid": 3,
                    },
                    "clone": {
                        "name": "clone",
                        "oid": 4,
                        "relfilenode": 5,
                        "heap_oid": 3,
                    },
                },
                "comparison": comparison,
            }
        )
        self.assertEqual(
            stable["comparison"]["left_bfs_locality"],
            comparison["left_bfs_locality"],
        )
        self.assertEqual(
            stable["comparison"]["right_bfs_locality"],
            comparison["right_bfs_locality"],
        )

    def test_v3_proof_binds_complete_heap_coverage_and_edge_span(self) -> None:
        comparison = {
            "format": "sqlens-hnsw-compare-v3",
            "same_heap": True,
            "logical_equal": True,
            "physical_equal": False,
            "entry_equal": True,
            "definition_equal": True,
            "tuple_coverage_equal": True,
            "left_nodes": 5,
            "right_nodes": 5,
            "left_heap_tids": 6,
            "right_heap_tids": 6,
            "left_tombstones": 0,
            "right_tombstones": 0,
            "left_definition_digest": "sha256:" + "1" * 64,
            "right_definition_digest": "sha256:" + "1" * 64,
            "left_tuple_coverage_digest": "sha256:" + "2" * 64,
            "right_tuple_coverage_digest": "sha256:" + "2" * 64,
            "left_logical_digest": "sha256:" + "3" * 64,
            "right_logical_digest": "sha256:" + "3" * 64,
            "left_physical_digest": "sha256:" + "4" * 64,
            "right_physical_digest": "sha256:" + "5" * 64,
            "left_bfs_locality": locality([1, 2, 2, 3, 4]),
            "right_bfs_locality": locality([8, 8, 9, 9, 10]),
            "left_edge_span": edge_span(5),
            "right_edge_span": edge_span(5),
        }
        proof = {
            "source_index": "source",
            "clone_index": "clone",
            "relations": {
                "source": {"name": "source", "oid": 1, "relfilenode": 2, "heap_oid": 3},
                "clone": {"name": "clone", "oid": 4, "relfilenode": 5, "heap_oid": 3},
            },
            "comparison": comparison,
        }
        validated = benchmark.validate_d2_graph_proof(
            proof, "source", "clone", expected_heap_tids=6
        )
        self.assertEqual(
            validated["proof_contract"],
            "sqlens_same_heap_same_logical_graph_physical_layout_v3",
        )
        self.assertEqual(validated["comparison"]["left_heap_tids"], 6)
        with self.assertRaisesRegex(RuntimeError, "expected candidate rows"):
            benchmark.validate_d2_graph_proof(
                proof, "source", "clone", expected_heap_tids=5
            )

    def test_contract_is_bounded_sample_but_full_statistics(self) -> None:
        source = (ROOT.parent.parent / "third_party/pgvector-sqlens/src/hnswclone.c").read_text()
        smoke = (ROOT / "sql/pgvector_clone_formality_smoke.sql").read_text()
        for marker in (
            "sqlens-hnsw-bfs-locality-v1",
            "HNSW_BFS_LOCALITY_SAMPLE_LIMIT 256",
            "sameOrNextPagePairs",
            "nondecreasingPairs",
            "full_statistics",
            "rank_samples",
        ):
            self.assertIn(marker, source)
        self.assertIn("source/clone BFS locality comparison is not symmetric", smoke)


if __name__ == "__main__":
    unittest.main()
