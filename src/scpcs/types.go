package scpcs

import (
	"fmt"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type Instance struct {
	NumElements   int
	NumSubsets    int
	Subsets       *mat.Dense
	Costs         *mat.VecDense
	Conflicts     *mat.Dense
	ConflictsList [][]int
}

type Solution struct {
	SelectedSubsets *mat.VecDense
	TotalCost       float64
}

type Node struct {
	CurrentSolution *Solution
	DualBound       float64
	FixedSubsets    int
	LagrangeanMul   *mat.VecDense
}

func (sol *Solution) String() string {
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

func (sol *Node) String() string {
	s := new(strings.Builder)
	fmt.Fprintln(s, "Fixed subsets:", sol.FixedSubsets)
	s.WriteString("Selected subsets: [ ")
	for i := 0; i < sol.CurrentSolution.SelectedSubsets.Len(); i++ {
		if sol.CurrentSolution.SelectedSubsets.AtVec(i) > 0.5 {
			fmt.Fprint(s, i)
			s.WriteString(" ")
		}
	}
	s.WriteString("]\n")
	fmt.Fprint(s, "Dual bound: ", sol.DualBound)
	return s.String()
}

func (inst *Instance) String() string {
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

	s.WriteString("Conflicts:\n")
	for _, pair := range inst.ConflictsList {
		s.WriteString(fmt.Sprintf("%v\tCost: %f\n", pair, inst.Conflicts.At(pair[0], pair[1])))
	}

	return s.String()
}
