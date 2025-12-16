#!/usr/bin/env python3
"""
Example with Auto-tuning: Beacon Bracket Query for TP53 - Use case 1
========================================================
This script automatically searches for CKKS parameters that work for the TP53 use case.
NOT all parameters work and we need to find the correct combination of multDepth, scaleModSize and scaleSignFHEW.
"""

import openfhe as fhe

print("="*70)
print("Beacon V2 Bracket Query - TP53 with Parameter Auto-tuning")
print("="*70)

# ============================================================================
# REAL DATA: TP53 Gene on Chromosome 17
# ============================================================================
print("\nGenomic Context:")
print("   Gene: TP53 (tumor suppressor)")
print("   Chromosome: 17")
print("   Coordinates: chr17:7,668,421-7,687,490 (GRCh38)")
print("   Relevance: Mutated in >50% of cancers")

# ============================================================================
# PROBLEM DATA
# ============================================================================
# Researcher's query (Client)
query_start_min = 5_000_000
query_start_max = 7_676_592
query_end_min = 7_669_607
query_end_max = 10_000_000

print("\nResearcher's Beacon Query:")
print("   Search: Focal deletions overlapping with TP53")
print(f"   start: [{query_start_min:,}, {query_start_max:,}]")
print(f"   end:   [{query_end_min:,}, {query_end_max:,}]")

# Variant in database (Server)
variant_start = 7_100_000
variant_end = 8_300_000

print("\nPatient Variant:")
print(f"   Region: chr17:{variant_start:,}-{variant_end:,}")
print(f"   Size: {(variant_end - variant_start)/1_000_000:.1f} Mb")
print(f"   Type: DEL (deletion)")

# Manual verification
check1 = query_start_min <= variant_start <= query_start_max
check2 = query_end_min <= variant_end <= query_end_max

print("\nManual Verification:")
print(f"   Expected match: {check1 and check2}")

# Expected result for FHE
expected = [-1, 1, -1, 1]

# ============================================================================
# PARAMETERS TO TEST
# ============================================================================
print("\n" + "="*70)
print("Optimal CKKS Parameter Search")
print("="*70)

# Fixed parameters
firstModSize = 60
slots = 4

# Parameters to test
multDepth_candidates = [12, 14, 16, 18]
scaleModSize_candidates = [40, 45, 50, 52, 54, 56, 58, 59]
scaleSignFHEW_candidates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 
                           1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

print(f"\nTesting combinations:")
print(f"  multDepth: {multDepth_candidates}")
print(f"  scaleModSize: {scaleModSize_candidates}")
print(f"  scaleSignFHEW: {len(scaleSignFHEW_candidates)} values (1e-2 to 1e-15)")
print(f"\nTotal configurations: {len(multDepth_candidates) * len(scaleModSize_candidates) * len(scaleSignFHEW_candidates)}")

# ============================================================================
# SEARCH FOR WORKING CONFIGURATION
# ============================================================================
config_found = False
attempts = 0
successful_configs = []
failed_configs = []

print("\nStarting search...\n")

for multDepth in multDepth_candidates:
    for scaleModSize in scaleModSize_candidates:
        for scaleSignFHEW in scaleSignFHEW_candidates:
            attempts += 1
            
            print(f"[{attempts:3d}] Testing: mD={multDepth}, sMS={scaleModSize}, scale={scaleSignFHEW:.0e}", end=" ... ")
            
            try:
                # ============================================================
                # Setup crypto context
                # ============================================================
                parameters = fhe.CCParamsCKKSRNS()
                parameters.SetMultiplicativeDepth(multDepth)
                parameters.SetScalingModSize(scaleModSize)
                parameters.SetFirstModSize(firstModSize)
                parameters.SetScalingTechnique(fhe.FLEXIBLEAUTOEXT)
                parameters.SetSecurityLevel(fhe.HEStd_128_classic)
                parameters.SetBatchSize(slots)
                parameters.SetKeySwitchTechnique(fhe.HYBRID)
                parameters.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
                
                cc = fhe.GenCryptoContext(parameters)
                cc.Enable(fhe.PKE)
                cc.Enable(fhe.KEYSWITCH)
                cc.Enable(fhe.LEVELEDSHE)
                cc.Enable(fhe.ADVANCEDSHE)
                cc.Enable(fhe.SCHEMESWITCH)
                
                ringDim = cc.GetRingDimension()
                if ringDim > 65536:
                    print(f"FAIL: ringDim={ringDim} too large")
                    failed_configs.append((multDepth, scaleModSize, scaleSignFHEW, "ringDim too large"))
                    continue
                
                # ============================================================
                # Generate keys
                # ============================================================
                keys = cc.KeyGen()
                
                # Setup scheme switching
                params = fhe.SchSwchParams()
                params.SetSecurityLevelCKKS(fhe.HEStd_128_classic)
                params.SetSecurityLevelFHEW(fhe.STD128)
                params.SetCtxtModSizeFHEWLargePrec(25)
                params.SetNumSlotsCKKS(slots)
                params.SetNumValues(slots)
                
                privateKeyFHEW = cc.EvalSchemeSwitchingSetup(params)
                ccLWE = cc.GetBinCCForSchemeSwitch()
                cc.EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW)
                
                modulus_LWE = 1 << 25
                beta = ccLWE.GetBeta()
                pLWE2 = int(modulus_LWE / (2 * beta))
                
                # ============================================================
                # Encrypt
                # ============================================================
                x1 = [query_start_min, query_start_max, query_end_min, query_end_max]
                ptxt1 = cc.MakeCKKSPackedPlaintext(x1, 1, 0, None, slots)
                c1 = cc.Encrypt(keys.publicKey, ptxt1)
                
                x2 = [variant_start, variant_start, variant_end, variant_end]
                ptxt2 = cc.MakeCKKSPackedPlaintext(x2, 1, 0, None, slots)
                c2 = cc.Encrypt(keys.publicKey, ptxt2)
                
                # ============================================================
                # Compare
                # ============================================================
                cc.EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW)
                cResult = cc.EvalCompareSchemeSwitching(c1, c2, slots, slots)
                
                # ============================================================
                # Decrypt and verify
                # ============================================================
                result = cc.Decrypt(keys.secretKey, cResult)
                result.SetLength(slots)
                vals = result.GetRealPackedValue()
                
                # Round to -1 or 1
                eps = 0.01
                rounded = [1 if round(v / eps) * eps == 0 else -1 for v in vals]
                
                if rounded == expected:
                    print(f"SUCCESS! (ringDim={ringDim})")
                    successful_configs.append({
                        'multDepth': multDepth,
                        'scaleModSize': scaleModSize,
                        'scaleSignFHEW': scaleSignFHEW,
                        'ringDim': ringDim,
                        'result': rounded
                    })
                    config_found = True
                    # Don't break here to find ALL working configurations
                else:
                    print(f"FAIL: result={rounded} != expected={expected}")
                    failed_configs.append((multDepth, scaleModSize, scaleSignFHEW, f"wrong result: {rounded}"))
                    
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 50:
                    error_msg = error_msg[:50] + "..."
                print(f"ERROR: {error_msg}")
                failed_configs.append((multDepth, scaleModSize, scaleSignFHEW, error_msg))

# ============================================================================
# SEARCH RESULTS
# ============================================================================
print("\n" + "="*70)
print("Parameter Search Summary")
print("="*70)

print(f"\nStatistics:")
print(f"   Configurations tested: {attempts}")
print(f"   Successful: {len(successful_configs)}")
print(f"   Failed: {len(failed_configs)}")
print(f"   Success rate: {len(successful_configs)/attempts*100:.1f}%")

if successful_configs:
    print(f"\nConfigurations that WORK ({len(successful_configs)}):")
    print("   " + "-"*66)
    for i, config in enumerate(successful_configs, 1):
        print(f"   [{i}] multDepth={config['multDepth']}, "
              f"scaleModSize={config['scaleModSize']}, "
              f"scaleSignFHEW={config['scaleSignFHEW']:.0e}")
        print(f"       ringDim={config['ringDim']}, result={config['result']}")
    
    # Show details of first successful configuration
    print("\n" + "="*70)
    print("Detailed Execution with First Successful Configuration")
    print("="*70)
    
    best_config = successful_configs[0]
    multDepth = best_config['multDepth']
    scaleModSize = best_config['scaleModSize']
    scaleSignFHEW = best_config['scaleSignFHEW']
    
    print(f"\nSelected Parameters:")
    print(f"   Multiplicative Depth: {multDepth}")
    print(f"   Scale Modulus Size: {scaleModSize}")
    print(f"   Scale FHEW: {scaleSignFHEW:.0e}")
    print(f"   Slots: {slots}")
    
    # Re-run with successful configuration to show details
    print("\nConfiguring crypto context...")
    parameters = fhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(multDepth)
    parameters.SetScalingModSize(scaleModSize)
    parameters.SetFirstModSize(firstModSize)
    parameters.SetScalingTechnique(fhe.FLEXIBLEAUTOEXT)
    parameters.SetSecurityLevel(fhe.HEStd_128_classic)
    parameters.SetBatchSize(slots)
    parameters.SetKeySwitchTechnique(fhe.HYBRID)
    parameters.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    
    cc = fhe.GenCryptoContext(parameters)
    cc.Enable(fhe.PKE)
    cc.Enable(fhe.KEYSWITCH)
    cc.Enable(fhe.LEVELEDSHE)
    cc.Enable(fhe.ADVANCEDSHE)
    cc.Enable(fhe.SCHEMESWITCH)
    
    print(f"   Ring Dimension: {cc.GetRingDimension()}")
    print("\nGenerating FHE keys...")
    keys = cc.KeyGen()
    
    params = fhe.SchSwchParams()
    params.SetSecurityLevelCKKS(fhe.HEStd_128_classic)
    params.SetSecurityLevelFHEW(fhe.STD128)
    params.SetCtxtModSizeFHEWLargePrec(25)
    params.SetNumSlotsCKKS(slots)
    params.SetNumValues(slots)
    
    privateKeyFHEW = cc.EvalSchemeSwitchingSetup(params)
    ccLWE = cc.GetBinCCForSchemeSwitch()
    cc.EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW)
    
    modulus_LWE = 1 << 25
    beta = ccLWE.GetBeta()
    pLWE2 = int(modulus_LWE / (2 * beta))
    
    print("   Public Key, Secret Key, Evaluation Keys generated")
    
    print("\n" + "="*70)
    print("STEP 1: Client encrypts their query")
    print("="*70)
    
    x1 = [query_start_min, query_start_max, query_end_min, query_end_max]
    print(f"\nQuery (plaintext): {x1}")
    
    ptxt1 = cc.MakeCKKSPackedPlaintext(x1, 1, 0, None, slots)
    c1 = cc.Encrypt(keys.publicKey, ptxt1)
    
    print("Query encrypted with CKKS")
    
    print("\n" + "="*70)
    print("STEP 2: Server encrypts their variant")
    print("="*70)
    
    x2 = [variant_start, variant_start, variant_end, variant_end]
    print(f"\nVariant (plaintext): {x2}")
    
    ptxt2 = cc.MakeCKKSPackedPlaintext(x2, 1, 0, None, slots)
    c2 = cc.Encrypt(keys.publicKey, ptxt2)
    
    print("Variant encrypted with client's publicKey")
    
    print("\n" + "="*70)
    print("STEP 3: Server performs homomorphic comparisons")
    print("="*70)
    
    cc.EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW)
    print(f"\nComparing c1 vs c2 (both encrypted)...")
    print("Operation: sign(c1 - c2)")
    
    cResult = cc.EvalCompareSchemeSwitching(c1, c2, slots, slots)
    
    print("Comparison completed (result still encrypted)")
    
    print("\n" + "="*70)
    print("STEP 4: Client decrypts result")
    print("="*70)
    
    result = cc.Decrypt(keys.secretKey, cResult)
    result.SetLength(slots)
    vals = result.GetRealPackedValue()
    
    print(f"\nDecrypted values (raw): {[f'{v:.6f}' for v in vals]}")
    
    eps = 0.01
    rounded = [1 if round(v / eps) * eps == 0 else -1 for v in vals]
    
    print(f"Rounded signs:          {rounded}")
    
    print("\n" + "="*70)
    print("Results Interpretation")
    print("="*70)
    
    print(f"\nExpected result for MATCH: {expected}")
    print(f"Obtained result:           {rounded}")
    
    comparisons = [
        ("start_min <= variant_start", query_start_min, variant_start, rounded[0], -1),
        ("variant_start <= start_max", variant_start, query_start_max, rounded[1], 1),
        ("end_min <= variant_end", query_end_min, variant_end, rounded[2], -1),
        ("variant_end <= end_max", variant_end, query_end_max, rounded[3], 1)
    ]
    
    print("\nComparison verification:")
    for desc, a, b, got, exp in comparisons:
        status = "OK" if got == exp else "FAIL"
        print(f"  [{status}] {desc}")
        print(f"     {a:,} vs {b:,} -> sign={got} (expected {exp})")
    
    if rounded == expected:
        print("\n" + "="*70)
        print("SUCCESS: MATCH detected correctly")
        print("The variant overlaps with TP53 and was found with total privacy!")
        print("="*70)
    
    print("\n" + "="*70)

else:
    print("\nNo working configuration found")

# Show some failed configurations for educational purposes
if failed_configs and len(failed_configs) <= 20:
    print(f"\nExamples of FAILED configurations (first {min(10, len(failed_configs))}):")
    for i, (md, sms, scale, error) in enumerate(failed_configs[:10], 1):
        print(f"   [{i}] mD={md}, sMS={sms}, scale={scale:.0e}")
        print(f"       Error: {error}")

print("\n" + "="*70)
print("Auto-tuning completed")
print("="*70)