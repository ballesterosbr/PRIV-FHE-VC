package pir

import (
	"fmt"
	"math/rand"
)

// BeaconVariant - Equivalente exacto a tu clase Python
type BeaconVariant struct {
	Chrom       string
	Pos         uint64
	Ref         string // Puede ser "" para DEL/INS
	Alt         string // Puede ser "" para DEL/INS
	VariantType string // "SNP", "DEL", "INS", etc.
}

func (v BeaconVariant) String() string {
	if v.Ref != "" && v.Alt != "" {
		return fmt.Sprintf("%s:%d %s>%s (%s)", v.Chrom, v.Pos, v.Ref, v.Alt, v.VariantType)
	}
	return fmt.Sprintf("%s:%d %s", v.Chrom, v.Pos, v.VariantType)
}

// EncodeVariant - Traducción EXACTA de tu función Python
func EncodeVariant(ref, alt, variantType string) uint64 {
	baseCodes := map[string]uint64{
		"A": 1, "C": 2, "G": 3, "T": 4,
	}
	typeCodes := map[string]uint64{
		"SNP": 0, "INDEL": 100, "DEL": 200, "DUP": 300, "CNV": 400, "INS": 500,
	}

	code := typeCodes[variantType]

	if ref != "" && alt != "" {
		code += baseCodes[ref] * 10
		code += baseCodes[alt]
	}

	return code
}

// DecodeVariant - Traducción EXACTA de tu función Python
func DecodeVariant(code uint64) string {
	if code == 0 {
		return "No variant"
	}

	typeCodes := map[string]uint64{
		"SNP": 0, "INDEL": 100, "DEL": 200, "DUP": 300, "CNV": 400, "INS": 500,
	}
	baseCodes := map[uint64]string{
		1: "A", 2: "C", 3: "G", 4: "T",
	}

	// Identificar tipo de variante
	variantType := ""
	basePart := uint64(0)

	for vtype, vcode := range typeCodes {
		if code >= vcode && code < vcode+100 {
			variantType = vtype
			basePart = code - vcode
			break
		}
	}

	if variantType == "SNP" && basePart > 0 {
		refCode := basePart / 10
		altCode := basePart % 10
		ref := baseCodes[refCode]
		alt := baseCodes[altCode]
		return fmt.Sprintf("%s>%s (%s)", ref, alt, variantType)
	}

	if variantType != "" {
		return variantType
	}
	return "Unknown"
}

// SyntheticVariantsChr17
func SyntheticVariantsChr17(n uint64) []BeaconVariant {
	bases := []string{"A", "C", "G", "T"}
	variants := make([]BeaconVariant, n)

	for pos := uint64(0); pos < n; pos++ {
		// random.choices con weights [0.9, 0.07, 0.03]
		r := rand.Float64()
		var vtype string

		if r < 0.90 { // 90% SNP
			vtype = "SNP"
		} else if r < 0.97 { // 7% DEL
			vtype = "DEL"
		} else { // 3% INS
			vtype = "INS"
		}

		if vtype == "SNP" {
			ref := bases[rand.Intn(4)]

			// Elegir alt != ref
			altBases := make([]string, 0, 3)
			for _, b := range bases {
				if b != ref {
					altBases = append(altBases, b)
				}
			}
			alt := altBases[rand.Intn(len(altBases))]

			variants[pos] = BeaconVariant{"chr17", pos, ref, alt, "SNP"}

		} else if vtype == "DEL" {
			variants[pos] = BeaconVariant{"chr17", pos, "", "", "DEL"}

		} else { // INS
			variants[pos] = BeaconVariant{"chr17", pos, "", "", "INS"}
		}

		if (pos+1)%10_000_000 == 0 {
			fmt.Printf("  Generadas %dM variantes...\n", (pos+1)/1_000_000)
		}
	}

	return variants
}

// CreateBeaconDatabase - Traducción EXACTA de tu función Python
func CreateBeaconDatabase(variants []BeaconVariant, genomeSize uint64) []uint64 {
	db := make([]uint64, genomeSize)

	fmt.Println("Procesando variantes...")
	count := 0

	for _, variant := range variants {
		if variant.Pos < genomeSize {
			code := EncodeVariant(variant.Ref, variant.Alt, variant.VariantType)
			db[variant.Pos] = code
			count++

			// Progress cada millón
			if count%10_000_000 == 0 {
				fmt.Printf("  Procesadas %d variantes...\n", count)
			}
		}
	}

	fmt.Printf("Total procesadas: %d\n", count)
	return db
}

// CountNonZero - Equivalente a np.count_nonzero
func CountNonZero(db []uint64) int {
	count := 0
	for _, val := range db {
		if val != 0 {
			count++
		}
	}
	return count
}
