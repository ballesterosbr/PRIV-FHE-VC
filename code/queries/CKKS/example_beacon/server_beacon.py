"""
Server-side Operations for Beacon V2 FHE Bracket Queries
========================================================

The server (genomic database owner) performs these operations:
1. Receives crypto context and public key from client
2. Encrypts its variant data using client's public key
3. Performs homomorphic comparisons using evaluation keys
4. Returns encrypted results to client (without learning the query or results)

PRIVACY GUARANTEES:
------------------
- Server NEVER sees the client's query coordinates (encrypted in c1)
- Server NEVER sees the comparison results (encrypted output)
- Client NEVER sees server's variant data in plaintext (encrypted in c2)
- Computation happens on encrypted data throughout

WHAT SERVER HAS ACCESS TO:
-------------------------
YES Crypto context (cc) with evaluation keys
YES Public key (for encrypting its own data)
YES Encrypted query from client (c1)
NO Secret key (only client has this)
NO Plaintext query coordinates
NO Decrypted comparison results
"""

import openfhe as fhe
import time
from psutil import Process
import os


def get_memory_usage_mb():
    """Monitor server RAM usage during homomorphic operations"""
    return Process(os.getpid()).memory_info().rss / 1024 / 1024


def prepare_server_ciphertext(cc, publicKey, variant_start, variant_end, slots):
    """
    Server encrypts its variant coordinates for comparison
    
    For Beacon bracket query, the server needs to encrypt the variant's
    start and end positions in a format compatible with the client's query.
    
    Format: [variant_start, variant_start, variant_end, variant_end]
    This allows 4 comparisons against client's [start_min, start_max, end_min, end_max]
    
    The variant coordinates are encrypted with the CLIENT'S public key,
    ensuring only the client can decrypt the final comparison results.
    
    Args:
        cc: Crypto context (received from client)
        publicKey: Client's public key
        variant_start: Start position of the variant (e.g., deletion start)
        variant_end: End position of the variant (e.g., deletion end)
        slots: Number of slots (should be 4 for bracket query)
    
    Returns:
        c2: Encrypted variant coordinates
    """
    x2 = [variant_start, variant_start, variant_end, variant_end]
    ptxt2 = cc.MakeCKKSPackedPlaintext(x2, 1, 0, None, slots)
    c2 = cc.Encrypt(publicKey, ptxt2)
    return c2


def evaluate_comparison(cc, c1, c2, slots, pLWE2, scaleSignFHEW, reps=20):
    """
    Perform homomorphic comparisons for Beacon bracket query matching
    
    This is the core FHE operation where the server computes:
        sign(c1 - c2)
    
    Without decrypting anything, giving:
        1. sign(start_min - variant_start)
        2. sign(start_max - variant_start)
        3. sign(end_min - variant_end)
        4. sign(end_max - variant_end)
    
    For a MATCH, we expect: [-1, 1, -1, 1]
    This means:
        - start_min ≤ variant_start (comparison 1 gives -1)
        - variant_start ≤ start_max (comparison 2 gives 1)
        - end_min ≤ variant_end (comparison 3 gives -1)
        - variant_end ≤ end_max (comparison 4 gives 1)
    
    Technical details:
        - EvalCompareSwitchPrecompute: Prepares comparison circuits
        - EvalCompareSchemeSwitching: Performs CKKS→FHEW→CKKS conversion
          * CKKS is good for arithmetic but not comparisons
          * FHEW is good for comparisons (sign extraction)
          * Results are converted back to CKKS for client decryption
    
    Args:
        cc: Crypto context with evaluation keys
        c1: Client's encrypted query [start_min, start_max, end_min, end_max]
        c2: Server's encrypted variant [variant_start, variant_start, variant_end, variant_end]
        slots: Number of comparisons (4)
        pLWE2: FHEW parameter
        scaleSignFHEW: Scale for sign evaluation
        reps: Number of repetitions for benchmarking
    
    Returns:
        results: List of encrypted comparison results
        eval_times: Time for each comparison operation
        server_ram_mb: RAM usage during computation
    """
    # Precompute comparison circuits for efficiency
    cc.EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW)
    
    results = []
    eval_times = []

    mem_before = get_memory_usage_mb()

    for _ in range(reps):
        start = time.perf_counter()
        
        # Homomorphic comparison: compute sign(c1 - c2) on encrypted data
        # This is the magic of FHE - comparison without seeing the numbers!
        cResult = cc.EvalCompareSchemeSwitching(c1, c2, slots, slots)
        
        eval_times.append(time.perf_counter() - start)
        results.append(cResult)
    
    mem_after = get_memory_usage_mb()
    server_ram_mb = mem_after - mem_before

    return results, eval_times, server_ram_mb
