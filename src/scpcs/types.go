package main

import (
	"fmt"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type SCPCSInstance struct {
	NumElements int
	NumSubsets  int
	Subsets     *mat.Dense
	Costs       *mat.VecDense
	Conflicts   *mat.Dense
}

type SCPCSSolution struct {
	SelectedSubsets *mat.VecDense
	TotalCost       float64
}

type SCPCSPartialSolution struct {
	FixedSubsets    *mat.VecDense
	SelectedSubsets *mat.VecDense
	TotalCost       float64
}

func (sol *SCPCSSolution) String() string {
	s := new(strings.Builder)
	s.WriteString(fmt.Sprintf("Total cost: %f\n", sol.TotalCost))
	s.WriteString("Selected subsets: [ ")
	for i := 0; i < sol.SelectedSubsets.Len(); i++ {
		if sol.SelectedSubsets.AtVec(i) > 0.5 {
			s.WriteString(fmt.Sprint(i))
			s.WriteString(" ")
		}
	}
	s.WriteString("]")
	return s.String()
}

func (inst *SCPCSInstance) String() string {
	s := new(strings.Builder)
	s.WriteString(fmt.Sprintf("N. elements: %d\n", inst.NumElements))
	s.WriteString(fmt.Sprintf("N. sets: %d\n", inst.Subsets.RawMatrix().Cols))

	for i := range inst.Subsets.RawMatrix().Cols {
		s.WriteString(fmt.Sprintf("Cost: %f, ", inst.Costs.At(i, 0)))
		s.WriteString("Elements: ")
		for e := range inst.NumElements {
			if inst.Subsets.At(e, i) == 1 {
				s.WriteString(fmt.Sprintf("%d ", e))
			}
		}
		s.WriteRune('\n')
	}

	return s.String()
}
