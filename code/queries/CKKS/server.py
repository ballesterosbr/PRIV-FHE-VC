import openfhe as fhe
import time
from psutil import Process
import os


def get_memory_usage_mb():
    return Process(os.getpid()).memory_info().rss / 1024 / 1024


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


# def setup_scheme_switching(cc, slots, logQ_ccLWE=25):
#     sl = fhe.HEStd_128_classic
#     slBin = fhe.STD128

#     params = fhe.SchSwchParams()
#     params.SetSecurityLevelCKKS(sl)
#     params.SetSecurityLevelFHEW(slBin)
#     params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE)
#     params.SetNumSlotsCKKS(slots)
#     params.SetNumValues(slots)

#     privateKeyFHEW = cc.EvalSchemeSwitchingSetup(params)
#     ccLWE = cc.GetBinCCForSchemeSwitch()

#     modulus_LWE = 1 << logQ_ccLWE
#     beta = ccLWE.GetBeta()
#     pLWE2 = int(modulus_LWE / (2 * beta))

#     return pLWE2


def prepare_server_ciphertext(cc, publicKey, secret_start, secret_end, slots):
    x2 = [secret_start, secret_start, secret_end, secret_end]
    ptxt2 = cc.MakeCKKSPackedPlaintext(x2, 1, 0, None, slots)
    c2 = cc.Encrypt(publicKey, ptxt2)
    return c2


def evaluate_comparison(cc, c1, c2, slots, pLWE2, scaleSignFHEW, reps=20):
    cc.EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW)
    
    results = []
    eval_times = []

    mem_before = get_memory_usage_mb()

    for _ in range(reps):
        start = time.perf_counter()
        cResult = cc.EvalCompareSchemeSwitching(c1, c2, slots, slots)
        eval_times.append(time.perf_counter() - start)
        results.append(cResult)
    
    mem_after = get_memory_usage_mb()
    server_ram_mb = mem_after - mem_before

    return results, eval_times, server_ram_mb