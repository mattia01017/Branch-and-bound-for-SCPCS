package generator

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
)

func GenerateSCPInstance(numSubsets, numElements int, meanDensity, stdDevDensity float64) string {
	s := new(strings.Builder)
	fmt.Fprintf(s, "%d %d\n", numElements, numSubsets)
	for range numSubsets {
		fmt.Fprintf(s, "%d ", 1+rand.Intn(20))
	}
	s.WriteRune('\n')

	for range numElements {
		r := math.Max(0, math.Min(1, meanDensity+stdDevDensity*rand.NormFloat64()))
		setSize := int(math.Max(1.0, float64(numSubsets)*r))
		p := rand.Perm(numSubsets)
		fmt.Println(p)
		fmt.Fprintf(s, "%d ", setSize)
		for i := range setSize {
			fmt.Fprintf(s, "%d ", p[i]+1)
		}
		s.WriteRune('\n')
	}
	return s.String()
}

func main() {
	// GenerateSCPInstance(50, 50)
	os.WriteFile("data/test3.txt", []byte(GenerateSCPInstance(50, 50, 0.2, 0.05)), 0666)
}
