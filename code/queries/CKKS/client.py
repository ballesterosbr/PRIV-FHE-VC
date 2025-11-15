import openfhe as fhe
import time
import math
import psutil

def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def setup_crypto_context(multDepth, scaleModSize, firstModSize, slots, secretKeyDist):
    sl = fhe.HEStd_128_classic

    parameters = fhe.CCParamsCKKSRNS()
    parameters.SetMultiplicativeDepth(multDepth)
    parameters.SetScalingModSize(scaleModSize)
    parameters.SetFirstModSize(firstModSize)
    parameters.SetScalingTechnique(fhe.FLEXIBLEAUTOEXT)
    parameters.SetSecurityLevel(sl)
    parameters.SetBatchSize(slots)
    parameters.SetKeySwitchTechnique(fhe.HYBRID)
    parameters.SetSecretKeyDist(getattr(fhe, secretKeyDist))

    cc = fhe.GenCryptoContext(parameters)
    cc.Enable(fhe.PKE)
    cc.Enable(fhe.KEYSWITCH)
    cc.Enable(fhe.LEVELEDSHE)
    cc.Enable(fhe.ADVANCEDSHE)
    cc.Enable(fhe.SCHEMESWITCH)

    return cc


def generate_keys_and_switching(cc, slots, logQ_ccLWE=25):
    start_keygen = time.perf_counter()

    mem_before_keygen = get_memory_usage_mb()
    keys = cc.KeyGen()
    mem_after_keygen = get_memory_usage_mb()

    keygen_time = time.perf_counter() - start_keygen

    mem_keygen_mb = mem_after_keygen - mem_before_keygen

    # FHEW switching setup
    params = fhe.SchSwchParams()
    params.SetSecurityLevelCKKS(fhe.HEStd_128_classic)
    params.SetSecurityLevelFHEW(fhe.STD128)
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE)
    params.SetNumSlotsCKKS(slots)
    params.SetNumValues(slots)

    privateKeyFHEW = cc.EvalSchemeSwitchingSetup(params)
    ccLWE = cc.GetBinCCForSchemeSwitch()
    cc.EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW)

    modulus_LWE = 1 << logQ_ccLWE
    beta = ccLWE.GetBeta()
    pLWE2 = int(modulus_LWE / (2 * beta))

    return keys, keygen_time, mem_keygen_mb, pLWE2


def encrypt_vector(cc, keys, x, slots):
    ptxt = cc.MakeCKKSPackedPlaintext(x, 1, 0, None, slots)
    start_encrypt = time.perf_counter()
    ctxt = cc.Encrypt(keys.publicKey, ptxt)
    encrypt_time = time.perf_counter() - start_encrypt
    return ctxt, encrypt_time


def decrypt_and_evaluate(cc, secretKey, cResults, expected_result, slots, eps=0.01):
    decrypt_times = []
    correct_count = 0
    fail_count = 0

    for cResult in cResults:
        start_decrypt = time.perf_counter()
        result = cc.Decrypt(secretKey, cResult)
        decrypt_times.append(time.perf_counter() - start_decrypt)

        result.SetLength(slots)
        vals = result.GetRealPackedValue()
        rounded = [1 if round(v / eps) * eps == 0 else -1 for v in vals]

        if rounded == expected_result:
            correct_count += 1
        else:
            fail_count += 1

    return correct_count, fail_count, decrypt_times


def client_pipeline(test_case, x1, x2, expected_result, multDepth, scaleModSize,
                    slots, secretKeyDist, scaleSignFHEW, reps):
    firstModSize = 60
    logQ_est = firstModSize + multDepth * (scaleModSize + 1)

    cc = setup_crypto_context(multDepth, scaleModSize, firstModSize, slots, secretKeyDist)

    ringDim = cc.GetRingDimension()
    if ringDim > 65536:
        raise RuntimeError(f"ringDim too big: {ringDim}")

    keys, keygen_time, mem_keygen_mb, pLWE2 = generate_keys_and_switching(cc, slots)
    
    mem_before_encrypt = get_memory_usage_mb()
    c1, encrypt1_time = encrypt_vector(cc, keys, x1, slots)
    c2, encrypt2_time = encrypt_vector(cc, keys, x2, slots)
    mem_after_encrypt = get_memory_usage_mb()

    mem_encrypt_mb = mem_after_encrypt - mem_before_encrypt
    encrypt_time_avg = (encrypt1_time + encrypt2_time) / 2

    try:
        cc.EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW)
        scaleSignFHEW_valid = scaleSignFHEW
    except Exception as e:
        raise RuntimeError(f"Scale {scaleSignFHEW} is not valid: {e}")

    return {
        "test_case": test_case,
        "multDepth": multDepth,
        "scaleModSize": scaleModSize,
        "firstModSize": firstModSize,
        "slots": slots,
        "keySwitchTechnique": "HYBRID",
        "secretKeyDist": secretKeyDist,
        "ringDim": ringDim,
        "logQ_estimated": logQ_est,
        "logQ_actual": math.log2(cc.GetModulus()),
        "keygen_time": keygen_time,
        "encrypt_time": encrypt_time_avg,
        "scaleSignFHEW": scaleSignFHEW_valid,
        "cc": cc,
        "publicKey": keys.publicKey,
        "secretKey": keys.secretKey,
        "c1": c1,
        "pLWE2": pLWE2,
        "expected_result": expected_result,
        "mem_keygen_mb": mem_keygen_mb,
        "mem_encrypt_mb": mem_encrypt_mb
    }
