# Private Genomic Queries with SimplePIR

Privacy-preserving variant lookups on chromosome 17 using Private Information Retrieval (PIR).

## Credits

**This implementation is built entirely on [SimplePIR](https://github.com/ahenzinger/simplepir) by Alexandra Henzinger, Matthew M. Hong, Henry Corrigan-Gibbs, Sarah Meiklejohn, and Vinod Vaikuntanathan (USENIX Security 2023).**

All cryptographic primitives, PIR protocols, and performance optimizations come from their work. Please visit the [original repository](https://github.com/ahenzinger/simplepir) for the full PIR library and cite their paper:
```bibtex
@inproceedings{cryptoeprint:2022/949,
  author = {Alexandra Henzinger and Matthew M. Hong and Henry Corrigan-Gibbs 
            and Sarah Meiklejohn and Vinod Vaikuntanathan},
  title = {One Server for the Price of Two: Simple and Fast Single-Server 
           Private Information Retrieval},
  booktitle = {32nd USENIX Security Symposium (USENIX Security 23)},
  year = {2023},
}
```

## What This Benchmark Does

Adapts DoublePIR for **Beacon V2 genomic queries** on chromosome 17 (83,257,441 base pairs). The client privately queries specific genomic positions without revealing which variant they're looking for. The server responds without learning the query position.

**Study goal**: Evaluate PIR overhead for real-world genomic databases.

## Use Case

- **Database**: Chromosome 17 variants (SNPs, insertions, deletions)
- **Query**: "Does position X contain a variant?"
- **Privacy**: Server doesn't know which position was queried

## Requirements

- **Go** ≥ 1.19.1 ([install](https://go.dev/))
- **GCC** or compatible C compiler
- **SimplePIR library** (cloned in this directory)

## Setup

The SimplePIR library should be cloned at the same level:
```bash
# From the queries/ directory
git clone https://github.com/ahenzinger/simplepir.git
```

## Usage

### Quick Test (5 strategic positions)
```bash
go run beacon_benchmark.go -quick
```

Queries 5 predefined positions at different scales: 100, 1K, 100K, 1M, and 45M. Tests PIR performance across the chromosome's range.

### Full Benchmark (11 positions)
```bash
go run beacon_benchmark.go -full
```

Sweeps chromosome 17 from start to end: 100, 1K, 10K, 100K, 1M, 10M, 45M, 50M, 80M, 83,257,440 (last valid), 84,257,440 (out-of-bounds, expected failure). Validates PIR correctness across the entire genomic range and boundary conditions.

### Random Queries
```bash
go run beacon_benchmark.go -random=20 -seed=42
```

Queries 20 **uniformly random positions** within chr17's 83 million base pairs. Simulates realistic researcher behavior. The seed makes results reproducible.

### Default (3 positions)
```bash
go run beacon_benchmark.go
```

Minimal test with 3 mid-range positions (1M, 10M, 45M). Quick sanity check.

## Command-Line Flags

- `-quick`: Run 5 strategic queries across chromosome scale
- `-full`: Run complete 11-query suite (start to end + boundary test)
- `-random=N`: Query N uniformly distributed random positions
- `-seed=X`: Set random seed (0 = use timestamp)
- `-synthetic`: Use realistic variant distribution (90% SNP, 7% DEL, 3% INS) [default: true]

## Output

Results are printed to console as a formatted table and saved to `benchmark_results_<timestamp>.csv` with columns: position, variant type, timings (query/answer/recovery), bandwidth metrics, and success status.

## Database Details

### Synthetic Variant Generation

The `-synthetic` flag (default) creates a **realistic genomic database** for chr17:

- **90% SNPs** (single nucleotide polymorphisms): Most common variant type
- **7% Deletions**: Small sequence removals
- **3% Insertions**: Small sequence additions
- **Overall density**: ~10% of positions contain variants (matching real genomic data)

Variants are encoded as 32-bit values with type information. Position 0 = no variant.

When `-synthetic=false`, a random 10% density database is used instead (uniform distribution, no biological structure).

### Technical Specifications

- **Size**: 83,257,441 positions (chr17 length in GRCh38)
- **Encoding**: 32-bit integers per position
- **Security**: N = 1024, log(q) = 32 (DoublePIR parameters)
- **Protocol**: Offline hint (16 MB) + online query/answer (~590 KB total per query)

## Command-Line Flags

- `-quick`: Run 5 strategic queries
- `-full`: Run complete 11-query suite
- `-random=N`: Query N random positions
- `-seed=X`: Set random seed (0 = use timestamp)
- `-synthetic`: Use realistic variant distribution (90% SNP, 7% DEL, 3% INS) [default: true]