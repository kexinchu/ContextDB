-- FragReuse smoke for ChooseKind / Admit. Empty development database only.
-- Load the library before any hnsw.* SET so the GUCs register.
SELECT vector_sqlens_build_id() = 'sqlens-v19-fragreuse-admit-20260830'
	AS admit_build;

SET enable_seqscan = off;
SET hnsw.ef_search = 64;
SET hnsw.filter_strategy = safe_guided;
SET hnsw.d3_probe_requests = 2;

DROP TABLE IF EXISTS fragreuse_admit_t;
CREATE TABLE fragreuse_admit_t (
	id int,
	cat text,
	helpful int,
	val vector(2)
);
INSERT INTO fragreuse_admit_t
SELECT i,
	CASE WHEN i % 10 = 0 THEN 'g' ELSE 'o' END,
	CASE WHEN i % 4 = 0 THEN 1 ELSE 0 END,
	format('[%s,%s]', i, i + 1)::vector
FROM generate_series(1, 400) AS g(i);
CREATE INDEX fragreuse_admit_idx ON fragreuse_admit_t USING hnsw (val vector_l2_ops)
WITH (m = 8, ef_construction = 32);
ANALYZE fragreuse_admit_t;
SELECT vector_hnsw_fragment_tracking_enable('fragreuse_admit_t'::regclass) IS NOT NULL
	AS tracking_on;

-- Adaptive first, while the fragment cache is empty.
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:cat = ''g'''],
	'adaptive'
) = 0 AS adaptive_probe_1;
SELECT count(*) = 30 AS adaptive_q1
FROM (
	SELECT id FROM fragreuse_admit_t
	WHERE (SELECT vector_hnsw_guidance_bind(
			'fragreuse_admit_idx'::regclass,
			ARRAY['sql:cat = ''g'''],
			'adaptive'
		) OFFSET 0)
	  AND cat = 'g'
	ORDER BY val <-> '[10,11]'::vector
	LIMIT 30
) AS q1;
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:cat = ''g'''],
	'adaptive'
) = 0 AS adaptive_probe_2;
SELECT count(*) = 30 AS adaptive_q2
FROM (
	SELECT id FROM fragreuse_admit_t
	WHERE (SELECT vector_hnsw_guidance_bind(
			'fragreuse_admit_idx'::regclass,
			ARRAY['sql:cat = ''g'''],
			'adaptive'
		) OFFSET 0)
	  AND cat = 'g'
	ORDER BY val <-> '[20,21]'::vector
	LIMIT 30
) AS q2;
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:cat = ''g'''],
	'adaptive'
) > 0 AS selective_atom_admitted;
SELECT vector_hnsw_guidance_profile()::jsonb->>'adaptive_state'
	IN ('exact', 'page', 'bloom') AS selective_state_live;
SELECT vector_hnsw_guidance_profile()::jsonb->>'kind' IN ('exact', 'page', 'bloom')
	AS selective_kind_bound;
SELECT (vector_hnsw_guidance_profile()::jsonb->>'kind') <> 'off'
	AND (vector_hnsw_guidance_profile()::jsonb->>'atoms')::int = 1
	AS selective_one_atom;

-- A tautology adds no skip once the selective atom is resident.
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:id >= 1'],
	'adaptive'
) = 0 AS tautology_probe_1;
SELECT count(*) = 30 AS tautology_q1
FROM (
	SELECT id FROM fragreuse_admit_t
	WHERE (SELECT vector_hnsw_guidance_bind(
			'fragreuse_admit_idx'::regclass,
			ARRAY['sql:id >= 1'],
			'adaptive'
		) OFFSET 0)
	ORDER BY val <-> '[10,11]'::vector
	LIMIT 30
) AS t1;
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:id >= 1'],
	'adaptive'
) = 0 AS tautology_probe_2;
SELECT count(*) = 30 AS tautology_q2
FROM (
	SELECT id FROM fragreuse_admit_t
	WHERE (SELECT vector_hnsw_guidance_bind(
			'fragreuse_admit_idx'::regclass,
			ARRAY['sql:id >= 1'],
			'adaptive'
		) OFFSET 0)
	ORDER BY val <-> '[20,21]'::vector
	LIMIT 30
) AS t2;
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:id >= 1'],
	'adaptive'
) = 0 AS tautology_rejected;
SELECT vector_hnsw_guidance_profile()::jsonb->>'adaptive_state' = 'rejected'
	AS tautology_state_rejected;

-- Named page on two atoms must not install a page bitmap.
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:cat = ''g''', 'sql:helpful >= 1'],
	'page'
) AS two_atom_page_activate;
SELECT vector_hnsw_guidance_profile()::jsonb->>'kind' = 'bloom'
	AS two_atom_page_became_bloom;
SELECT (vector_hnsw_guidance_profile()::jsonb->>'page_compose_upgraded')::boolean
	AS page_compose_upgraded;
SELECT (vector_hnsw_guidance_profile()::jsonb->>'atoms')::int = 2
	AS two_atoms_bound;
SELECT count(*) > 0 AS guided_query_runs
FROM (
	SELECT id FROM fragreuse_admit_t
	WHERE (SELECT vector_hnsw_guidance_bind(
			'fragreuse_admit_idx'::regclass,
			ARRAY['sql:cat = ''g''', 'sql:helpful >= 1'],
			'page'
		) OFFSET 0)
	  AND cat = 'g'
	  AND helpful >= 1
	ORDER BY val <-> '[10,11]'::vector
	LIMIT 5
) AS q;

SET hnsw.d3_compose_complete_only = off;
SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:cat = ''g''', 'sql:helpful >= 1'],
	'page'
) AS two_atom_page_legacy;
SELECT vector_hnsw_guidance_profile()::jsonb->>'kind' = 'page'
	AS legacy_page_compose_allowed;
RESET hnsw.d3_compose_complete_only;

SELECT vector_hnsw_guidance_reset();
SELECT vector_hnsw_guidance_activate(
	'fragreuse_admit_idx'::regclass,
	ARRAY['sql:helpful >= 1'],
	'page'
) AS unary_page_activate;
SELECT vector_hnsw_guidance_profile()::jsonb->>'kind' = 'page'
	AS unary_page_stays_page;
