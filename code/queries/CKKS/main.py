import os
import gc
import pandas as pd
from client import client_pipeline, get_memory_usage_mb, decrypt_and_evaluate
from server import (
    # setup_scheme_switching,
    prepare_server_ciphertext,
    evaluate_comparison
)

def clear_memory():
    gc.collect()


# Global settings
REPS = 20
CSV_FILE = "openfhe_benchmark_results.csv"

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

# Test cases
test_cases = {
    "1": {
        "description": "Small values",
        "secret_start": 7_100_000,
        "secret_end":   8_300_000,
        "start_min":    5_000_000,
        "start_max":    7_676_592,
        "end_min":      7_669_607,
        "end_max":      10_000_000
    },
    "2": {
        "description": "Medium values",
        "secret_start": 12_345_678_901_234,
        "secret_end":   12_345_679_900_000,
        "start_min":    12_345_000_000_000,
        "start_max":    12_345_700_000_000,
        "end_min":      12_345_600_000_000,
        "end_max":      12_345_800_000_000
    },
    "3": {
        "description": "Large values",
        "secret_start": 15_984_726_351_478_902_345,
        "secret_end":   15_984_726_751_478_902_345,
        "start_min":    15_984_726_000_000_000_000,
        "start_max":    15_984_726_500_000_000_000,
        "end_min":      15_984_726_700_000_000_000,
        "end_max":      15_984_727_000_000_000_000
    }
}

# Parameters
multDepth_list = [12, 14, 16, 18]
# scaleModSize_list = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 59]
scaleModSize_list = [50, 52, 54, 56, 58, 59]
secretKeyDist_list = ["UNIFORM_TERNARY"]
slots_list = [4]
# test_case_list = [1, 2, 3]
test_case_list = [3]
scale_candidates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15] # up to 1e-15 (test case 3)

# Execution loop
for test_case_num in test_case_list:
    case = test_cases[str(test_case_num)]
    x1 = [case['start_min'], case['start_max'], case['end_min'], case['end_max']]
    expected_result = [-1 if a - b < 0 else 1 for a, b in zip(x1, [case['secret_start'], case['secret_start'], case['secret_end'], case['secret_end']])]
    secret_start = case["secret_start"]
    secret_end = case["secret_end"]

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

                            # Client: generate keys, context, encrypt vectors, decrypt results
                            client_result = client_pipeline(
                                test_case=test_case_num,
                                x1=x1,
                                x2=[secret_start, secret_start, secret_end, secret_end],
                                expected_result=expected_result,
                                multDepth=multDepth,
                                scaleModSize=scaleModSize,
                                slots=slots,
                                secretKeyDist=secretKeyDist,
                                scaleSignFHEW=scaleSignFHEW,
                                reps=REPS
                            )

                            # Server: Use cc and client's publicKey
                            cc = client_result["cc"]
                            publicKey = client_result["publicKey"]
                            c1 = client_result["c1"]

                            # Server: generates c2 (secret) and evaluates comparisons
                            pLWE2 = client_result["pLWE2"]
                            c2 = prepare_server_ciphertext(cc, publicKey, secret_start, secret_end, slots)

                            eval_results, eval_times, server_ram_mb = evaluate_comparison(
                                cc, c1, c2, slots, pLWE2, client_result["scaleSignFHEW"], reps=REPS
                            )

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

                            print(f"Computed — compare_time_avg: {client_result['compare_time_avg']:.2f}s")
                            print(f"RAM usage: {get_memory_usage_mb()} MB")

                        except Exception as e:
                            print(f"ERROR in config {config_key}: {str(e)}")
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

print("\nBenchmark completed. Results stored in:", CSV_FILE)