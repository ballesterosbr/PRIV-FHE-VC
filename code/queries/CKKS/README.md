# CKKS Homomorphic Encryption for Beacon V2 Bracket Queries

Privacy-preserving genomic variant queries using CKKS fully homomorphic encryption (FHE). CKKS encodes genomic coordinates as approximate real numbers, enabling homomorphic arithmetic. Comparisons use CKKS -> FHEW scheme switching to extract signs without decryption.

## What it does

Implements Beacon V2 bracket queries with FHE to allow researchers to search genomic databases without revealing their query coordinates. The server performs homomorphic comparisons without learning the query or results.

## Quick Start

### Requirements
```bash
uv sync
```

### Run benchmark
```bash
uv run main.py
```

Benchmarks different CKKS parameter combinations and saves results to `openfhe_benchmark_results.csv`.

### Example: TP53 Use Case

The `example_beacon/` folder contains two ready-to-run examples:
```bash
cd example_beacon

# Parameter search (finds working CKKS configurations)
uv run tp53_autotuning.py

# Simple example with fixed parameters
uv run tp53_simple.py
```

**Scenario**: Oncologist searches for patients with focal deletions affecting TP53 tumor suppressor gene (chr17:7,668,421-7,687,490).

- **Query** (encrypted): Deletion ranges that would overlap TP53
- **Variant** (encrypted): Patient deletion chr17:7,100,000-8,300,000
- **Result**: MATCH detected with full privacy

**Note**: `tp53_autotuning.py` systematically tests CKKS parameter combinations to find configurations that work for this specific query. Not all parameter combinations are valid. Use `tp53_simple.py` to see the protocol in action with pre-validated parameters.

## Files

- `client.py` - Client operations (key generation, encryption, decryption)
- `server.py` - Server operations (homomorphic comparisons)
- `main.py` - Benchmark harness testing parameter combinations
- `example_beacon/` - Real-world TP53 examples