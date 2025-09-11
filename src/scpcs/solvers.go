package main

import (
	"github.com/lukpank/go-glpk/glpk"
)

func arange(start, end int) []int32 {
	nums := make([]int32, end-start)
	startI32 := int32(start)
	for i := range int32(end - start) {
		nums[i] = startI32 + i
	}
	return nums
}

func runMIPSolver(prob *glpk.Prob) (*SCPCSSolution, error) {
	iocp := glpk.NewIocp()
	iocp.SetPresolve(true)
	iocp.SetMsgLev(glpk.MSG_OFF)

	if err := prob.Intopt(iocp); err != nil {
		return nil, err
	}

	selected := make([]int, 0)
	for j := range prob.NumCols() {
		if prob.MipColVal(j+1) > 0.5 {
			selected = append(selected, j)
		}
	}

	return &SCPCSSolution{
		SelectedSubsets: selected,
		TotalCost:       int(prob.MipObjVal()),
	}, nil
}

func (inst *SCPCSInstance) defSCPCS() *glpk.Prob {
	prob := glpk.New()
	prob.SetObjDir(glpk.MIN)

	numVars := len(inst.Subsets)
	prob.AddCols(numVars)
	for j := range numVars {
		prob.SetColKind(j+1, glpk.BV)
		prob.SetObjCoef(j+1, float64(inst.Subsets[j].Cost))
	}

	numConstraints := inst.NumElements
	prob.AddRows(numConstraints)
	for i := range numConstraints {
		prob.SetRowBnds(i+1, glpk.LO, 1, 1)
	}

	mask := make([]float64, numConstraints+1)
	for j, subset := range inst.Subsets {
		for i := range numConstraints {
			if subset.Set.Contains(int32(i + 1)) {
				mask[i+1] = 1
			} else {
				mask[i+1] = 0
			}
		}
		prob.SetMatCol(j+1, arange(0, numConstraints+1), mask)
	}

	mask = make([]float64, numVars+1)
	for i := range len(inst.Conflicts) - 1 {
		for j := i + 1; j < len(inst.Conflicts); j++ {
			if inst.Conflicts[i][j] == 1 {
				prob.AddCols(1)
				prob.AddRows(1)
				numVars++
				numConstraints++
				prob.SetRowBnds(numConstraints, glpk.UP, 1, 1)

				for k := range mask {
					mask[k] = 0
				}
				mask[i+1] = 1
				mask[j+1] = 1
				mask = append(mask, -1)
				prob.SetMatRow(numConstraints, arange(0, numVars+1), mask)
			}
		}

	}
	return prob
}

func (inst *SCPCSInstance) Solve() (*SCPCSSolution, error) {
	prob := inst.defSCPCS()
	return runMIPSolver(prob)
}
