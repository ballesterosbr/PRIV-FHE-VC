"""
Beacon V2 Bracket Query Implementation using CKKS Homomorphic Encryption
=========================================================================

This code implements privacy-preserving genomic variant queries using the 
Beacon V2 protocol's bracket query mechanism with fully homomorphic encryption (FHE).

GENOMIC CONTEXT:
---------------
Beacon V2 allows geneticists to query databases for structural variants (CNVs) 
without revealing exact coordinates. The server responds true/false without 
learning what the researcher is looking for.

BRACKET QUERY LOGIC:
-------------------
A variant MATCHES the query if it falls within the specified genomic ranges:
    start_min ≤ variant_start ≤ start_max  AND
    end_min ≤ variant_end ≤ end_max

This is implemented via 4 homomorphic comparisons:
    1. start_min - variant_start ≤ 0  → result = -1
    2. start_max - variant_start ≥ 0  → result = +1
    3. end_min - variant_end ≤ 0      → result = -1
    4. end_max - variant_end ≥ 0      → result = +1

Expected result for MATCH: [-1, 1, -1, 1]

PRIVACY MODEL:
-------------
- Client encrypts query coordinates with their public key
- Server encrypts variant coordinates with client's public key
- Server performs homomorphic comparisons using evaluation keys
- Only client can decrypt final result (has secret key)
- Neither party learns the other's data during computation

REAL-WORLD USE CASE:
-------------------
Test Case 1 represents a real oncology scenario:

Gene: TP53 (tumor suppressor, chr17:7,668,421-7,687,490)
Query: Find patients with focal deletions affecting TP53
Beacon URL format:
    ?referenceName=chr17&variantType=DEL&start=5000000,7676592&end=7669607,10000000

This searches for deletions (DEL) that:
- Start anywhere from 2.5Mb upstream to just before TP53
- End anywhere from just after TP53 to 2.5Mb downstream
- Are focal (<5Mb), filtering out large chromosomal rearrangements

Example variant (patient with TP53 deletion):
- variant_start: 7,100,000 (starts before TP53)
- variant_end: 8,300,000 (ends after TP53)
- This patient MATCHES the query (deletion overlaps TP53)
"""

import os
import gc
import pandas as pd
from client import client_pipeline, get_memory_usage_mb, decrypt_and_evaluate
from server import prepare_server_ciphertext, evaluate_comparison

def clear_memory():
    gc.collect()


# Global settings
REPS = 20
CSV_FILE = "beacon_fhe_benchmark_results.csv"

# Load previous results
if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE)
    existing_configs = set(
        tuple(row)
        for row in df_existing[["test_case", "multDepth", "scaleModSize", "slots", "secretKeyDist", "scaleSignFHEW"]].to_numpy()
    )
else:
    df_existing = pd.DataFrame()
    existing_configs = set()

# Test cases representing different genomic scales
test_cases = {
    "1": {
        "description": "TP53 focal deletion (chr17) - Real Beacon example",
        "gene": "TP53",
        "chromosome": "chr17",
        "coordinates": "7,668,421-7,687,490 (GRCh38)",
        "variant_start": 7_100_000,    # Patient's deletion starts here
        "variant_end":   8_300_000,    # Patient's deletion ends here
        "start_min":     5_000_000,    # Query: allow variants starting from here
        "start_max":     7_676_592,    # Query: up to ~start of TP53
        "end_min":       7_669_607,    # Query: from ~end of TP53
        "end_max":       10_000_000,   # Query: up to here (focal <5Mb filter)
        "biological_meaning": "Focal deletion overlapping TP53 tumor suppressor gene"
    },
    "2": {
        "description": "Medium-scale genomic region (simulated)",
        "variant_start": 12_345_678_901_234,
        "variant_end":   12_345_679_900_000,
        "start_min":     12_345_000_000_000,
        "start_max":     12_345_700_000_000,
        "end_min":       12_345_600_000_000,
        "end_max":       12_345_800_000_000,
        "biological_meaning": "Testing larger coordinate values for scalability"
    },
    "3": {
        "description": "Large-scale genomic coordinates (stress test)",
        "variant_start": 15_984_726_351_478_902_345,
        "variant_end":   15_984_726_751_478_902_345,
        "start_min":     15_984_726_000_000_000_000,
        "start_max":     15_984_726_500_000_000_000,
        "end_min":       15_984_726_700_000_000_000,
        "end_max":       15_984_727_000_000_000_000,
        "biological_meaning": "Testing precision limits of CKKS encoding"
    }
}

# FHE Parameters for CKKS scheme
multDepth_list = [12, 14, 16, 18]
scaleModSize_list = [50, 52, 54, 56, 58, 59]
secretKeyDist_list = ["UNIFORM_TERNARY"]
slots_list = [4]  # 4 comparisons needed for bracket query
test_case_list = [1]  # Start with TP53 example
scale_candidates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

# Execution loop
for test_case_num in test_case_list:
    case = test_cases[str(test_case_num)]
    
    # Client's encrypted query (bracket ranges)
    x1 = [case['start_min'], case['start_max'], case['end_min'], case['end_max']]
    
    # Expected result for Bracket Query match: [-1, 1, -1, 1]
    # This verifies: start_min ≤ variant_start ≤ start_max AND end_min ≤ variant_end ≤ end_max
    expected_result = [-1 if a - b < 0 else 1 for a, b in zip(
        x1, 
        [case['variant_start'], case['variant_start'], case['variant_end'], case['variant_end']]
    )]
    
    variant_start = case["variant_start"]
    variant_end = case["variant_end"]
    
    print(f"\n{'='*80}")
    print(f"Test Case {test_case_num}: {case['description']}")
    if 'gene' in case:
        print(f"Gene: {case['gene']} ({case['coordinates']})")
    print(f"Variant region: [{variant_start:,} - {variant_end:,}]")
    print(f"Query ranges: start=[{case['start_min']:,}, {case['start_max']:,}], end=[{case['end_min']:,}, {case['end_max']:,}]")
    print(f"Expected match result: {expected_result}")
    print(f"Biological context: {case['biological_meaning']}")
    print(f"{'='*80}\n")

    for multDepth in multDepth_list:
        for scaleModSize in scaleModSize_list:
            for secretKeyDist in secretKeyDist_list:
                for slots in slots_list:
                    for scaleSignFHEW in scale_candidates:
                        config_key = (test_case_num, multDepth, scaleModSize, slots, secretKeyDist, scaleSignFHEW)
                        if config_key in existing_configs:
                            print(f"Skipping (already computed): {config_key}")
                            continue

                        try:
                            print(f"\nTesting: TC={test_case_num}, mD={multDepth}, sMS={scaleModSize}, slots={slots}, Dist={secretKeyDist}, scale={scaleSignFHEW}")

                            # STEP 1: Client setup
                            # - Generates crypto context and keys
                            # - Encrypts query coordinates
                            client_result = client_pipeline(
                                test_case=test_case_num,
                                x1=x1,  # Query ranges [start_min, start_max, end_min, end_max]
                                x2=[variant_start, variant_start, variant_end, variant_end],  # Variant coords
                                expected_result=expected_result,
                                multDepth=multDepth,
                                scaleModSize=scaleModSize,
                                slots=slots,
                                secretKeyDist=secretKeyDist,
                                scaleSignFHEW=scaleSignFHEW,
                                reps=REPS
                            )

                            # STEP 2: Server operations
                            # - Receives crypto context and public key from client
                            # - Encrypts its variant coordinates with client's public key
                            # - Performs homomorphic comparisons using evaluation keys
                            cc = client_result["cc"]
                            publicKey = client_result["publicKey"]
                            c1 = client_result["c1"]  # Client's encrypted query
                            pLWE2 = client_result["pLWE2"]

                            # Server encrypts its variant data
                            c2 = prepare_server_ciphertext(cc, publicKey, variant_start, variant_end, slots)

                            # Server performs homomorphic comparisons
                            eval_results, eval_times, server_ram_mb = evaluate_comparison(
                                cc, c1, c2, slots, pLWE2, client_result["scaleSignFHEW"], reps=REPS
                            )

                            # STEP 3: Client decrypts results
                            # - Only client has secret key
                            # - Decrypts comparison results to determine match/no-match
                            correct_count, fail_count, decrypt_times = decrypt_and_evaluate(
                                cc, 
                                client_result["secretKey"],
                                eval_results,
                                client_result["expected_result"],
                                slots
                            )

                            client_result["compare_time_avg"] = sum(eval_times) / len(eval_times)
                            client_result["decrypt_time_avg"] = sum(decrypt_times) / len(decrypt_times)
                            client_result["correct_count"] = correct_count
                            client_result["fail_count"] = fail_count

                            # Save results
                            exclude_keys = {"cc", "publicKey", "secretKey", "c1", "expected_result", "all_ckks_results"}
                            row_to_save = {k: v for k, v in client_result.items() if k not in exclude_keys}
                            
                            pd.DataFrame([row_to_save]).to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=False)

                            print(f"✓ Computed — compare_time_avg: {client_result['compare_time_avg']:.4f}s, "
                                  f"matches: {correct_count}/{REPS}")
                            print(f"  RAM usage: {get_memory_usage_mb():.1f} MB")

                        except Exception as e:
                            print(f"✗ ERROR in config {config_key}: {str(e)}")
                            error_row = {
                                "test_case": test_case_num,
                                "multDepth": multDepth,
                                "scaleModSize": scaleModSize,
                                "firstModSize": 60,
                                "slots": slots,
                                "keySwitchTechnique": "HYBRID",
                                "secretKeyDist": secretKeyDist,
                                "ringDim": None,
                                "logQ_estimated": None,
                                "logQ_actual": None,
                                "keygen_time": None,
                                "encrypt_time": None,
                                "compare_time_avg": None,
                                "decrypt_time_avg": None,
                                "correct_count": 0,
                                "fail_count": REPS,
                                "scaleSignFHEW": "",
                                "error": str(e)
                            }

                            pd.DataFrame([error_row]).to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=False)

                        finally:
                            clear_memory()

print(f"\n{'='*80}")
print("Beacon FHE Benchmark completed!")
print(f"Results stored in: {CSV_FILE}")
print(f"{'='*80}")
