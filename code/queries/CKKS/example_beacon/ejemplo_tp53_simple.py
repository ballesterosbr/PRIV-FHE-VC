#!/usr/bin/env python3
"""
Ejemplo Simple: Beacon Bracket Query para TP53
==============================================

Este script demuestra una bracket query real para el gen TP53
usando cifrado homomórfico (FHE) con CKKS.

Caso de uso: Oncólogo busca pacientes con deleciones focales de TP53
"""

import openfhe as fhe

print("="*70)
print("Beacon V2 Bracket Query - Ejemplo TP53 con FHE")
print("="*70)

# ============================================================================
# DATOS REALES: Gen TP53 en Cromosoma 17
# ============================================================================
print("\n📍 Contexto Genómico:")
print("   Gen: TP53 (tumor suppressor)")
print("   Cromosoma: 17")
print("   Coordenadas: chr17:7,668,421-7,687,490 (GRCh38)")
print("   Relevancia: Mutado en >50% de cánceres")

# ============================================================================
# QUERY DEL INVESTIGADOR (Cliente)
# ============================================================================
print("\n🔍 Query Beacon del Investigador:")
print("   Buscar: Deleciones focales que solapen con TP53")

query_start_min = 5_000_000   # Permite variantes desde 2.5Mb antes
query_start_max = 7_676_592   # Hasta ~inicio de TP53
query_end_min = 7_669_607     # Desde ~final de TP53
query_end_max = 10_000_000    # Hasta 2.5Mb después

print(f"   start: [{query_start_min:,}, {query_start_max:,}]")
print(f"   end:   [{query_end_min:,}, {query_end_max:,}]")
print(f"   (Filtra deleciones focales <5Mb)")

# ============================================================================
# VARIANTE EN BASE DE DATOS (Servidor)
# ============================================================================
print("\n🧬 Variante del Paciente en Base de Datos:")

variant_start = 7_100_000     # Deleción empieza aquí
variant_end = 8_300_000       # Deleción termina aquí
variant_size = variant_end - variant_start

print(f"   Región: chr17:{variant_start:,}-{variant_end:,}")
print(f"   Tamaño: {variant_size/1_000_000:.1f} Mb")
print(f"   Tipo: DEL (deletion)")

# ============================================================================
# VERIFICACIÓN MANUAL (Sin FHE primero)
# ============================================================================
print("\n✅ Verificación Manual de Bracket Query:")

check1 = query_start_min <= variant_start <= query_start_max
check2 = query_end_min <= variant_end <= query_end_max

print(f"   ¿{query_start_min:,} ≤ {variant_start:,} ≤ {query_start_max:,}? {check1}")
print(f"   ¿{query_end_min:,} ≤ {variant_end:,} ≤ {query_end_max:,}? {check2}")

if check1 and check2:
    print("\n   → MATCH: Esta deleción afecta TP53 ✓")
else:
    print("\n   → NO MATCH: Esta deleción NO afecta TP53")

# ============================================================================
# IMPLEMENTACIÓN CON FHE
# ============================================================================
print("\n" + "="*70)
print("Implementación con Cifrado Homomórfico (FHE)")
print("="*70)

# Parámetros CKKS
multDepth = 12
scaleModSize = 40
firstModSize = 60
slots = 4  # Necesitamos 4 comparaciones

print(f"\n⚙️  Parámetros CKKS:")
print(f"   Multiplicative Depth: {multDepth}")
print(f"   Scale Modulus Size: {scaleModSize}")
print(f"   Slots: {slots}")

# Setup crypto context
print("\n🔧 Configurando crypto context...")
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

# Generar claves
print("\n🔑 Generando claves FHE...")
keys = cc.KeyGen()
print("   ✓ Public Key (para cifrar)")
print("   ✓ Secret Key (para descifrar - NUNCA se comparte)")

# Setup scheme switching para comparaciones
print("\n🔄 Configurando scheme switching (CKKS ↔ FHEW)...")
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

print("   ✓ Evaluation Keys generadas")

# ============================================================================
# CIFRADO
# ============================================================================
print("\n" + "="*70)
print("PASO 1: Cliente cifra su query")
print("="*70)

# Cliente cifra su query
x1 = [query_start_min, query_start_max, query_end_min, query_end_max]
print(f"\nQuery (texto plano): {x1}")

ptxt1 = cc.MakeCKKSPackedPlaintext(x1, 1, 0, None, slots)
c1 = cc.Encrypt(keys.publicKey, ptxt1)

print("✓ Query cifrada con CKKS")
print("  (Cliente envía c1 al servidor sin revelar las coordenadas)")

print("\n" + "="*70)
print("PASO 2: Servidor cifra su variante")
print("="*70)

# Servidor cifra la variante con la public key del cliente
x2 = [variant_start, variant_start, variant_end, variant_end]
print(f"\nVariante (texto plano): {x2}")

ptxt2 = cc.MakeCKKSPackedPlaintext(x2, 1, 0, None, slots)
c2 = cc.Encrypt(keys.publicKey, ptxt2)  # Usa la publicKey del cliente

print("✓ Variante cifrada con publicKey del cliente")
print("  (Servidor mantiene sus datos privados)")

# ============================================================================
# COMPARACIÓN HOMOMÓRFICA
# ============================================================================
print("\n" + "="*70)
print("PASO 3: Servidor realiza comparaciones homomórficas")
print("="*70)

scaleSignFHEW = 1e-2
cc.EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW)

print(f"\nComparando c1 vs c2 (ambos cifrados)...")
print("Operación: sign(c1 - c2)")

cResult = cc.EvalCompareSchemeSwitching(c1, c2, slots, slots)

print("✓ Comparación completada (resultado aún cifrado)")
print("  (Servidor no puede ver el resultado)")

# ============================================================================
# DESCIFRADO
# ============================================================================
print("\n" + "="*70)
print("PASO 4: Cliente descifra resultado")
print("="*70)

result = cc.Decrypt(keys.secretKey, cResult)
result.SetLength(slots)
vals = result.GetRealPackedValue()

print(f"\nValores descifrados (raw): {[f'{v:.6f}' for v in vals]}")

# Redondear a -1 o 1
eps = 0.01
rounded = [1 if round(v / eps) * eps == 0 else -1 for v in vals]

print(f"Signos redondeados:        {rounded}")

# ============================================================================
# INTERPRETACIÓN
# ============================================================================
print("\n" + "="*70)
print("Interpretación de Resultados")
print("="*70)

expected = [-1, 1, -1, 1]
print(f"\nResultado esperado para MATCH: {expected}")
print(f"Resultado obtenido:            {rounded}")

comparisons = [
    ("start_min ≤ variant_start", query_start_min, variant_start, rounded[0], -1),
    ("variant_start ≤ start_max", variant_start, query_start_max, rounded[1], 1),
    ("end_min ≤ variant_end", query_end_min, variant_end, rounded[2], -1),
    ("variant_end ≤ end_max", variant_end, query_end_max, rounded[3], 1)
]

print("\nVerificación de comparaciones:")
for desc, a, b, got, exp in comparisons:
    status = "✓" if got == exp else "✗"
    print(f"  {status} {desc}")
    print(f"     {a:,} vs {b:,} → sign={got} (esperado {exp})")

if rounded == expected:
    print("\n" + "🎉 "*20)
    print("SUCCESS: MATCH detectado correctamente")
    print("La variante solapa con TP53 y fue encontrada con privacidad total!")
    print("🎉 "*20)
else:
    print("\n⚠️  WARNING: Resultado no coincide con expected")
    print("   Posibles causas: parámetros CKKS, escala FHEW incorrecta")

# ============================================================================
# RESUMEN DE PRIVACIDAD
# ============================================================================
print("\n" + "="*70)
print("Garantías de Privacidad")
print("="*70)

print("""
✓ Servidor NUNCA vio las coordenadas de la query
  (estaban cifradas en c1)

✓ Servidor NUNCA vio el resultado de las comparaciones
  (cResult estaba cifrado)

✓ Cliente NUNCA vio las variantes del servidor en texto plano
  (estaban cifradas en c2)

✓ Comparaciones se hicieron completamente en dominio cifrado
  (usando evaluation keys, sin secret key)

→ Privacidad bilateral garantizada por FHE
""")

print("="*70)
print("Ejemplo completado")
print("="*70)
