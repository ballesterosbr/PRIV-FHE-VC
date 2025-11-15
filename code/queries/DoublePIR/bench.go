package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/ahenzinger/simplepir/pir"
)

const BEACON_SEC_PARAM = uint64(1 << 10)
const BEACON_LOGQ = uint64(32)
const CHR17_SIZE = uint64(83_257_441)

type BenchmarkResult struct {
	QueryPosition    uint64
	ExpectedToFail   bool
	FoundVariant     bool
	VariantValue     uint64
	VariantDecoded   string
	SetupTimeMs      float64
	QueryTimeMs      float64
	AnswerTimeMs     float64
	RecoverTimeMs    float64
	TotalOnlineMs    float64
	OfflineKB        float64
	QueryUploadKB    float64
	AnswerDownloadKB float64
	TotalBandwidthKB float64
	Success          bool
	ActualResult     string
}

func (r BenchmarkResult) PrintRow() {
	variantStatus := "NO"
	if r.FoundVariant {
		variantStatus = "YES"
	}

	resultSymbol := "True"
	if r.ActualResult == "EXPECTED_FAIL" {
		resultSymbol = "True*"
	} else if !r.Success {
		resultSymbol = "False"
	}

	fmt.Printf("| %15d | %8s | %15s | %8.3f | %10.3f | %11.3f | %15.3f | %13.0f | %13.0f | %15.0f | %12.0f | %8s |\n",
		r.QueryPosition,
		variantStatus,
		r.VariantDecoded,
		r.QueryTimeMs,
		r.AnswerTimeMs,
		r.RecoverTimeMs,
		r.TotalOnlineMs,
		r.OfflineKB,
		r.QueryUploadKB,
		r.AnswerDownloadKB,
		r.TotalBandwidthKB,
		resultSymbol,
	)
}

func printHeader() {
	fmt.Println("\n" + strings.Repeat("=", 180))
	fmt.Printf("| %15s | %8s | %15s | %8s | %10s | %11s | %15s | %13s | %13s | %15s | %12s | %8s |\n",
		"Position", "Variant?", "Type", "Query(ms)", "Answer(ms)", "Recover(ms)", "TotalOnline(ms)",
		"Offline(KB)", "QueryUp(KB)", "AnswerDown(KB)", "TotalBW(KB)", "Result")
	fmt.Println(strings.Repeat("=", 180))
}

func printSeparator() {
	fmt.Println(strings.Repeat("-", 180))
}

type QuerySpec struct {
	Position       uint64
	ExpectedToFail bool
	Description    string
}

var globalDB []uint64
var globalDBGenerated = false

func ensureDBGenerated(synthetic bool) {
	if globalDBGenerated {
		return
	}

	fmt.Printf("\n=== GENERATING DATABASE (chr17: %d positions) ===\n", CHR17_SIZE)

	if synthetic {
		fmt.Println("Generating synthetic variants...")
		variants := pir.SyntheticVariantsChr17(CHR17_SIZE)
		globalDB = pir.CreateBeaconDatabase(variants, CHR17_SIZE)
		nonZero := pir.CountNonZero(globalDB)
		fmt.Printf("Stored variants: %d (%.2f%% density)\n",
			nonZero, float64(nonZero)/float64(CHR17_SIZE)*100)
	} else {
		fmt.Println("Generating random database...")
		globalDB = make([]uint64, CHR17_SIZE)
		for i := uint64(0); i < CHR17_SIZE; i++ {
			if rand.Float64() < 0.10 {
				globalDB[i] = uint64(rand.Int63n(1 << 32))
			}
		}
		nonZero := pir.CountNonZero(globalDB)
		fmt.Printf("Stored variants: %d (%.2f%% density)\n",
			nonZero, float64(nonZero)/float64(CHR17_SIZE)*100)
	}

	globalDBGenerated = true
	fmt.Println("Database ready\n")
}

func runSingleQuery(querySpec QuerySpec, DB *pir.Database, p pir.Params,
	serverState pir.State, serverShared pir.State, clientShared pir.State,
	offlineMsg pir.Msg, setupTimeMs float64, offlineKB float64, originalDB []uint64) BenchmarkResult {

	result := BenchmarkResult{
		QueryPosition:  querySpec.Position,
		ExpectedToFail: querySpec.ExpectedToFail,
		Success:        false,
		SetupTimeMs:    setupTimeMs,
		OfflineKB:      offlineKB,
	}

	pirImpl := &pir.DoublePIR{}

	if querySpec.Position >= CHR17_SIZE {
		result.ActualResult = "EXPECTED_FAIL"
		result.Success = querySpec.ExpectedToFail
		result.VariantDecoded = "Out of bounds"
		return result
	}

	defer func() {
		if r := recover(); r != nil {
			if querySpec.ExpectedToFail {
				result.ActualResult = "EXPECTED_FAIL"
				result.Success = true
				result.VariantDecoded = fmt.Sprintf("Failed as expected: %v", r)
			} else {
				result.ActualResult = "FAIL"
				result.Success = false
				result.VariantDecoded = fmt.Sprintf("Unexpected error: %v", r)
			}
		}
	}()

	t1 := time.Now()
	clientState, q := pirImpl.Query(querySpec.Position, clientShared, p, DB.Info)
	result.QueryTimeMs = float64(time.Since(t1).Microseconds()) / 1000.0
	result.QueryUploadKB = float64(q.Size()*uint64(p.Logq)) / (8.0 * 1024.0)

	t2 := time.Now()
	ans := pirImpl.Answer(DB, pir.MsgSlice{Data: []pir.Msg{q}}, serverState, serverShared, p)
	result.AnswerTimeMs = float64(time.Since(t2).Microseconds()) / 1000.0
	result.AnswerDownloadKB = float64(ans.Size()*uint64(p.Logq)) / (8.0 * 1024.0)

	t3 := time.Now()
	val := pirImpl.Recover(querySpec.Position, 0, offlineMsg, q, ans, clientShared, clientState, p, DB.Info)
	result.RecoverTimeMs = float64(time.Since(t3).Microseconds()) / 1000.0

	expected := originalDB[querySpec.Position]
	recoveredCorrectly := (expected == val)

	result.VariantValue = val
	result.FoundVariant = (val != 0)
	result.VariantDecoded = pir.DecodeVariant(val)

	if querySpec.ExpectedToFail {
		result.Success = false
		result.ActualResult = "FAIL"
	} else {
		result.Success = recoveredCorrectly
		if recoveredCorrectly {
			result.ActualResult = "PASS"
		} else {
			result.ActualResult = "FAIL"
		}
	}

	result.TotalOnlineMs = result.QueryTimeMs + result.AnswerTimeMs + result.RecoverTimeMs
	result.TotalBandwidthKB = result.OfflineKB + result.QueryUploadKB + result.AnswerDownloadKB

	return result
}

func main() {
	var (
		quick     = flag.Bool("quick", false, "Run only quick tests (few queries)")
		full      = flag.Bool("full", false, "Run complete suite with many queries")
		synthetic = flag.Bool("synthetic", true, "Use realistic synthetic variants")
		numRandom = flag.Int("random", 0, "Number of random positions to query")
		seed      = flag.Int64("seed", 0, "Seed for random generator (0 = use timestamp)")
	)
	flag.Parse()

	if *seed == 0 {
		rand.Seed(time.Now().UnixNano())
	} else {
		rand.Seed(*seed)
	}

	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("              BENCHMARK: Beacon PIR with DoublePIR (chr17)")
	fmt.Println(strings.Repeat("=", 80))

	fmt.Printf("\nSecurity parameters: N=%d, logq=%d\n", BEACON_SEC_PARAM, BEACON_LOGQ)
	fmt.Printf("DB size (chr17): %d positions\n", CHR17_SIZE)
	fmt.Printf("DB type: ")
	if *synthetic {
		fmt.Println("Synthetic variants (90% SNP, 7% DEL, 3% INS)")
	} else {
		fmt.Println("Random data (10% density)")
	}

	ensureDBGenerated(*synthetic)

	var querySpecs []QuerySpec

	if *numRandom > 0 {
		for i := 0; i < *numRandom; i++ {
			pos := uint64(rand.Int63n(int64(CHR17_SIZE)))
			querySpecs = append(querySpecs, QuerySpec{Position: pos, ExpectedToFail: false})
		}
	} else if *quick {
		querySpecs = []QuerySpec{
			{Position: 100, ExpectedToFail: false, Description: "Position near start"},
			{Position: 1_000, ExpectedToFail: false, Description: "Position 1K"},
			{Position: 100_000, ExpectedToFail: false, Description: "Position 100K"},
			{Position: 1_000_000, ExpectedToFail: false, Description: "Position 1M"},
			{Position: 45_000_001, ExpectedToFail: false, Description: "Standard test position"},
		}
	} else if *full {
		querySpecs = []QuerySpec{
			// {Position: 84_257_440, ExpectedToFail: true, Description: "Out of range (should fail)"},
			// {Position: 83_257_440, ExpectedToFail: false, Description: "Last valid position"},
			// {Position: 80_000_000, ExpectedToFail: false, Description: "80M"},
			// {Position: 50_000_000, ExpectedToFail: false, Description: "50M"},
			// {Position: 45_000_001, ExpectedToFail: false, Description: "Test position"},
			// {Position: 10_000_000, ExpectedToFail: false, Description: "10M"},
			// {Position: 1_000_000, ExpectedToFail: false, Description: "1M"},
			// {Position: 100_000, ExpectedToFail: false, Description: "100K"},
			// {Position: 10_000, ExpectedToFail: false, Description: "10K"},
			// {Position: 1_000, ExpectedToFail: false, Description: "1K"},
			// {Position: 100, ExpectedToFail: false, Description: "Start"},
			{Position: 100, ExpectedToFail: false, Description: "Start"},
			{Position: 1_000, ExpectedToFail: false, Description: "1K"},
			{Position: 10_000, ExpectedToFail: false, Description: "10K"},
			{Position: 100_000, ExpectedToFail: false, Description: "100K"},
			{Position: 1_000_000, ExpectedToFail: false, Description: "1M"},
			{Position: 10_000_000, ExpectedToFail: false, Description: "10M"},
			{Position: 45_000_001, ExpectedToFail: false, Description: "Test position"},
			{Position: 50_000_000, ExpectedToFail: false, Description: "50M"},
			{Position: 80_000_000, ExpectedToFail: false, Description: "80M"},
			{Position: 83_257_440, ExpectedToFail: false, Description: "Last valid position"},
			{Position: 84_257_440, ExpectedToFail: true, Description: "Out of range (should fail)"},
		}
	} else {
		querySpecs = []QuerySpec{
			{Position: 1_000_000, ExpectedToFail: false},
			{Position: 10_000_000, ExpectedToFail: false},
			{Position: 45_000_001, ExpectedToFail: false},
		}
	}

	fmt.Printf("\nQueries to execute: %d positions\n", len(querySpecs))

	fmt.Println("\n=== CONFIGURING PIR ===")
	pirImpl := &pir.DoublePIR{}
	p := pirImpl.PickParams(CHR17_SIZE, 32, BEACON_SEC_PARAM, BEACON_LOGQ)
	DB := pir.MakeDB(CHR17_SIZE, 32, &p, globalDB)

	seed_pir := pir.RandomPRGKey()
	serverShared, comp := pirImpl.InitCompressedSeeded(DB.Info, p, seed_pir)
	clientShared := pirImpl.DecompressState(DB.Info, p, comp)

	fmt.Println("\n=== SETUP (offline - one time) ===")
	t0 := time.Now()
	serverState, offlineMsg := pirImpl.Setup(DB, serverShared, p)
	setupTimeMs := float64(time.Since(t0).Microseconds()) / 1000.0
	offlineKB := float64(offlineMsg.Size()*uint64(p.Logq)) / (8.0 * 1024.0)

	fmt.Printf("Setup completed: %.2f ms\n", setupTimeMs)
	fmt.Printf("Offline download: %.2f KB\n", offlineKB)

	fmt.Println("\n=== EXECUTING QUERIES ===")
	var results []BenchmarkResult
	startTotal := time.Now()

	for i, spec := range querySpecs {
		expectedStr := ""
		if spec.ExpectedToFail {
			expectedStr = " (expected to fail)"
		}

		fmt.Printf("\n[%d/%d] Querying position %d%s...\n", i+1, len(querySpecs), spec.Position, expectedStr)
		if spec.Description != "" {
			fmt.Printf("  Description: %s\n", spec.Description)
		}

		result := runSingleQuery(spec, DB, p, serverState, serverShared, clientShared,
			offlineMsg, setupTimeMs, offlineKB, globalDB)
		results = append(results, result)

		if result.Success {
			if result.ExpectedToFail {
				fmt.Printf("  Failed as expected: %s\n", result.VariantDecoded)
			} else if result.FoundVariant {
				fmt.Printf("  Variant found: %s\n", result.VariantDecoded)
			} else {
				fmt.Printf("  No variant at this position\n")
			}
		} else {
			if result.ExpectedToFail {
				fmt.Printf("  Expected failure but succeeded\n")
			} else {
				fmt.Printf("  QUERY FAILED\n")
			}
		}
	}

	totalElapsed := time.Since(startTotal)
	pirImpl.Reset(DB, p)

	printHeader()
	for _, r := range results {
		r.PrintRow()
	}
	printSeparator()

	fmt.Printf("\n\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("                           RESULTS SUMMARY\n")
	fmt.Printf("%s\n\n", strings.Repeat("=", 80))

	successCount := 0
	variantsFound := 0
	expectedFails := 0
	var avgOnlineMs, avgBandwidthKB float64

	for _, r := range results {
		if r.Success {
			successCount++
		}
		if r.FoundVariant {
			variantsFound++
		}
		if r.ExpectedToFail && r.ActualResult == "EXPECTED_FAIL" {
			expectedFails++
		}
		avgOnlineMs += r.TotalOnlineMs
		avgBandwidthKB += r.TotalBandwidthKB
	}

	if len(results) > 0 {
		avgOnlineMs /= float64(len(results))
		avgBandwidthKB /= float64(len(results))
	}

	fmt.Printf("DB size (chr17):           %d positions\n", CHR17_SIZE)
	fmt.Printf("Queries executed:          %d\n", len(results))
	fmt.Printf("Successful queries:        %d\n", successCount)
	fmt.Printf("Expected failures (OK):    %d\n", expectedFails)
	fmt.Printf("Variants found:            %d\n", variantsFound)
	fmt.Printf("Success rate:              %.1f%%\n", float64(successCount)/float64(len(results))*100)
	fmt.Printf("\nSetup (offline, 1 time):   %.2f ms (%.2f KB)\n", setupTimeMs, offlineKB)
	fmt.Printf("Average per query:         %.2f ms\n", avgOnlineMs)
	fmt.Printf("Average BW per query:      %.2f KB\n", avgBandwidthKB-offlineKB)
	fmt.Printf("\nTotal query time:          %.2f seconds\n", totalElapsed.Seconds())

	csvFile := fmt.Sprintf("benchmark_results_%d.csv", time.Now().Unix())
	f, err := os.Create(csvFile)
	if err == nil {
		defer f.Close()
		fmt.Fprintf(f, "Position,ExpectedToFail,ActualResult,FoundVariant,VariantType,QueryMs,AnswerMs,RecoverMs,TotalOnlineMs,OfflineKB,QueryUpKB,AnswerDownKB,TotalBandwidthKB,Success\n")
		for _, r := range results {
			fmt.Fprintf(f, "%d,%v,%s,%v,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%v\n",
				r.QueryPosition, r.ExpectedToFail, r.ActualResult, r.FoundVariant, r.VariantDecoded,
				r.QueryTimeMs, r.AnswerTimeMs, r.RecoverTimeMs, r.TotalOnlineMs, r.OfflineKB,
				r.QueryUploadKB, r.AnswerDownloadKB, r.TotalBandwidthKB, r.Success)
		}
		fmt.Printf("\nResults saved to: %s\n", csvFile)
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("Benchmark completed.")
	fmt.Println(strings.Repeat("=", 80) + "\n")
}
