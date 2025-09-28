package main

import (
	"flag"
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
		fmt.Fprintf(s, "%d ", setSize)
		for i := range setSize {
			fmt.Fprintf(s, "%d ", p[i]+1)
		}
		s.WriteRune('\n')
	}
	return s.String()
}

func main() {
	var outPath string
	var numSubsets, numElements int
	var meanDensity, stdDevDensity float64

	flag.StringVar(&outPath, "out", "out.txt", "The output file")
	flag.IntVar(&numElements, "elems", 0, "The number of elements")
	flag.IntVar(&numSubsets, "sets", 0, "The number of subsets")
	flag.Float64Var(&meanDensity, "meand", 0, "The subsets density mean")
	flag.Float64Var(&stdDevDensity, "stddevd", 0, "The subsets density standard deviation")

	flag.Parse()

	err := false
	if numElements == 0 {
		fmt.Fprintln(os.Stderr, "Must specify the number of elements")
		err = true
	}
	if numSubsets == 0 {
		fmt.Fprintln(os.Stderr, "Must specify the number of elements")
		err = true
	}
	if meanDensity == 0 {
		fmt.Fprintln(os.Stderr, "Must specify subsets density mean")
		err = true
	}
	if stdDevDensity == 0 {
		fmt.Fprintln(os.Stderr, "Must specify subsets density standard deviation")
		err = true
	}

	if err {
		os.Exit(1)
	}

	os.WriteFile(
		outPath,
		[]byte(GenerateSCPInstance(numSubsets, numElements, meanDensity, stdDevDensity)),
		0666,
	)
}
