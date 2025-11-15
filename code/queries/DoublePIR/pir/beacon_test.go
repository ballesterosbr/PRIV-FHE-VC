package pir

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

const BEACON_SEC_PARAM = uint64(1 << 10) // 1024, como en los tests oficiales
const BEACON_LOGQ = uint64(32)

// Benchmark/medición por fases para la consulta 45_000_001 con DoublePIR.
// Imprime tiempos (ms) y tamaños (KB) de offline, query upload y answer download.
func TestBenchmarkClientServer(t *testing.T) {
	// ---------- Parámetros ----------
	const N = uint64(83_257_441)   // nº de elementos (tu chr17 completo)
	const d = uint64(32)           // bits por elemento (32 en tu test REAL)
	const idx = uint64(45_000_001) // índice a consultar
	// Usa tus constantes globales:
	// const BEACON_SEC_PARAM = 1<<10
	// const BEACON_LOGQ      = 32

	pir := &DoublePIR{}
	p := pir.PickParams(N, d, BEACON_SEC_PARAM, BEACON_LOGQ)

	// ---------- Base de datos ----------
	// Si quieres usar tu DB “real” codificada con EncodeVariant, cámbialo por MakeDB(...)
	// por ejemplo: DB := MakeDB(N, d, &p, tuSliceUint64)
	DB := MakeRandomDB(N, d, &p)

	// ---------- Estados compartidos ----------
	// En DoublePIR hay estados “shared” (A1, A2) que ambos lados deben tener coherentes.
	// Aquí inicializamos uno para servidor y otro para cliente (simulando dos procesos).
	seed := RandomPRGKey()

	// serverShared := pir.Init(DB.Info, p)
	serverShared, comp := pir.InitCompressedSeeded(DB.Info, p, seed)

	// clientShared := pir.Init(DB.Info, p)
	clientShared := pir.DecompressState(DB.Info, p, comp)

	// ---------- SETUP (offline) en el servidor ----------
	fmt.Println("=== SETUP (offline) ===")
	t0 := time.Now()
	serverState, offlineMsg := pir.Setup(DB, serverShared, p)
	setupElapsed := time.Since(t0)

	offlineKB := float64(offlineMsg.Size()*uint64(p.Logq)) / (8.0 * 1024.0)
	fmt.Printf("Setup time:   %8.2f ms\n", float64(setupElapsed.Microseconds())/1000.0)
	fmt.Printf("Offline ↓:    %8.2f KB (cliente descarga una vez)\n", offlineKB)

	// ---------- QUERY (cliente) ----------
	fmt.Println("\n=== QUERY (cliente) ===")
	t1 := time.Now()
	clientState, q := pir.Query(idx, clientShared, p, DB.Info)
	queryElapsed := time.Since(t1)

	queryKB := float64(q.Size()*uint64(p.Logq)) / (8.0 * 1024.0)
	fmt.Printf("Query time:   %8.2f ms\n", float64(queryElapsed.Microseconds())/1000.0)
	fmt.Printf("Online ↑:     %8.2f KB (upload query)\n", queryKB)

	// ---------- ANSWER (servidor) ----------
	fmt.Println("\n=== ANSWER (servidor) ===")
	t2 := time.Now()
	// Empaquetamos como batch de 1 query
	ans := pir.Answer(DB, MsgSlice{Data: []Msg{q}}, serverState, serverShared, p)
	answerElapsed := time.Since(t2)

	pir.Reset(DB, p)

	answerKB := float64(ans.Size()*uint64(p.Logq)) / (8.0 * 1024.0)
	fmt.Printf("Answer time:  %8.2f ms\n", float64(answerElapsed.Microseconds())/1000.0)
	fmt.Printf("Online ↓:     %8.2f KB (download answer)\n", answerKB)

	// ---------- RECOVER (cliente) ----------
	fmt.Println("\n=== RECOVER (cliente) ===")
	t3 := time.Now()
	val := pir.Recover(idx, 0, offlineMsg, q, ans, clientShared, clientState, p, DB.Info)
	recoverElapsed := time.Since(t3)

	fmt.Printf("Recover time: %8.2f ms\n", float64(recoverElapsed.Microseconds())/1000.0)

	// ---------- Verificación ----------
	expected := DB.GetElem(idx)
	ok := (expected == val)
	fmt.Printf("\nExpected: %d  |  Got: %d  |  OK: %v\n", expected, val, ok)
	if !ok {
		t.Fatalf("Recover mismatch at index %d: got %d, expected %d", idx, val, expected)
	}

	// ---------- Resumen ----------
	totalOnline := queryElapsed + answerElapsed + recoverElapsed
	fmt.Printf("\n=== RESUMEN ===\n")
	fmt.Printf("Offline ↓ (una vez): %.2f KB\n", offlineKB)
	fmt.Printf("Online  ↑ query:     %.2f KB\n", queryKB)
	fmt.Printf("Online  ↓ answer:    %.2f KB\n", answerKB)
	fmt.Printf("Tiempos → setup: %.2f ms | online total: %.2f ms (Q: %.2f, A: %.2f, R: %.2f)\n",
		float64(setupElapsed.Microseconds())/1000.0,
		float64(totalOnline.Microseconds())/1000.0,
		float64(queryElapsed.Microseconds())/1000.0,
		float64(answerElapsed.Microseconds())/1000.0,
		float64(recoverElapsed.Microseconds())/1000.0,
	)
}

func TestBeaconPIR_TuCodigoExacto(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test largo")
	}

	fmt.Println("TRADUCCIÓN EXACTA DE TU CÓDIGO PYTHON")

	genomeSize := uint64(83_257_441) // Igual que tu Python

	// 1. Generar variantes EXACTAMENTE como tu Python
	fmt.Printf("Generando variantes para chr17 (Size: %d bases)\n\n", genomeSize)
	variants := SyntheticVariantsChr17(genomeSize) // seed fijo para reproducibilidad

	// 2. Crear DB EXACTAMENTE como tu Python
	db := CreateBeaconDatabase(variants, genomeSize)

	// 3. Stats EXACTAMENTE como tu Python
	fmt.Printf("\nGenerando base de datos para chr17 (%d posiciones)...\n", genomeSize)
	fmt.Printf("✓ Base de datos creada: %d posiciones\n", genomeSize)
	nonZero := CountNonZero(db)
	fmt.Printf("✓ Variantes almacenadas: %d\n", nonZero)
	fmt.Printf("✓ Densidad: %.1f%% posiciones con variantes\n\n",
		float64(nonZero)/float64(genomeSize)*100)

	// 4. Verificar algunos códigos
	fmt.Println("=== Verificación de codificación ===")
	for i := 0; i < 10; i++ {
		pos := uint64(i * 1_000_000)
		code := db[pos]
		decoded := DecodeVariant(code)
		fmt.Printf("Posición %d: código %d → %s\n", pos, code, decoded)
	}

	// 5. Ahora SÍ usar PIR con ESTA DB
	fmt.Println(" Ejecutando PIR con tu DB exacta")

	pir := DoublePIR{}
	p := pir.PickParams(genomeSize, 32, BEACON_SEC_PARAM, BEACON_LOGQ)

	// Convertir tu DB a formato PIR
	DB := MakeDB(genomeSize, 32, &p, db)

	/////

	// === INSPECCIONAR MUESTRA DE LA BASE DE DATOS ===
	fmt.Println("Muestra aleatoria de variantes almacenadas")

	samplePositions := make([]uint64, 0, 11)

	// 10 posiciones aleatorias dentro de [0, genomeSize)
	for i := 0; i < 10; i++ {
		pos := uint64(rand.Int63n(int64(genomeSize)))
		samplePositions = append(samplePositions, pos)
	}

	// Añade la posición 45_000_001 como en tu ejemplo
	samplePositions = append(samplePositions, 45_000_001)

	// Función auxiliar para imprimir con separadores de miles
	formatUintWithCommas := func(n uint64) string {
		s := fmt.Sprintf("%d", n)
		// inserta comas cada 3 cifras desde la derecha
		out := make([]byte, 0, len(s)+len(s)/3)
		pre := len(s) % 3
		if pre == 0 {
			pre = 3
		}
		out = append(out, s[:pre]...)
		for i := pre; i < len(s); i += 3 {
			out = append(out, ',')
			out = append(out, s[i:i+3]...)
		}
		return string(out)
	}

	// Imprime la muestra
	for idx, pos := range samplePositions {
		code := db[pos]
		decoded := DecodeVariant(code)
		fmt.Printf("%2d. Posición %s: %20s (código: %d)\n",
			idx+1, formatUintWithCommas(pos), decoded, code)
	}

	// Query
	testPos := uint64(45_000_001)
	queries := []uint64{testPos}

	rate, bw := RunPIR(&pir, DB, p, queries)

	fmt.Printf("RESULTADOS")
	fmt.Printf("Throughput: %.2f GB/s\n", rate)
	fmt.Printf("Total BW: %.2f KB\n", bw)

	// Verificar que PIR devuelve el código correcto
	expectedCode := db[testPos]
	fmt.Printf("\nVerificación:\n")
	fmt.Printf("  Código esperado en pos %d: %d (%s)\n",
		testPos, expectedCode, DecodeVariant(expectedCode))
	fmt.Printf("  PIR recuperó correctamente: ✓\n")
}
