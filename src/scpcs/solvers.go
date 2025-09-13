package main

import (
	"github.com/lukpank/go-glpk/glpk"
	"gonum.org/v1/gonum/mat"
)

func arange(start, end int) []int32 {
	nums := make([]int32, end-start)
	startI32 := int32(start)
	for i := range int32(end - start) {
		nums[i] = startI32 + i
	}
	return nums
}

func (inst *SCPCSInstance) runMIPSolver(prob *glpk.Prob) (*SCPCSSolution, error) {
	iocp := glpk.NewIocp()
	iocp.SetPresolve(true)
	iocp.SetMsgLev(glpk.MSG_OFF)

	if err := prob.Intopt(iocp); err != nil {
		return nil, err
	}

	selected := mat.NewVecDense(inst.NumSubsets, nil)
	for j := range inst.NumSubsets {
		if prob.MipColVal(j+1) > 0.5 {
			selected.SetVec(j, 1)
		}
	}

	return &SCPCSSolution{
		SelectedSubsets: selected,
		TotalCost:       prob.MipObjVal(),
	}, nil
}

func (inst *SCPCSInstance) defBaseSCP(prob *glpk.Prob) {
	prob.SetObjDir(glpk.MIN)

	prob.AddCols(inst.NumSubsets)
	for j := range inst.NumSubsets {
		prob.SetColKind(j+1, glpk.BV)
		prob.SetObjCoef(j+1, float64(inst.Costs.At(j, 0)))
	}

	prob.AddRows(inst.NumElements)
	for i := range inst.NumElements {
		prob.SetRowBnds(i+1, glpk.LO, 1, 1)
	}

	colMask := make([]float64, prob.NumRows()+1)
	for j := range inst.NumElements {
		v := inst.Subsets.ColView(j)
		for i := range inst.NumElements {
			colMask[i+1] = v.At(i, 0)
		}
		prob.SetMatCol(j+1, arange(0, inst.NumElements+1), colMask)
	}
}

func (inst *SCPCSInstance) defConflicts(prob *glpk.Prob) {
	for i := range inst.NumSubsets - 1 {
		for j := i + 1; j < inst.NumSubsets; j++ {
			if inst.Conflicts.At(i, j) > 0 {
				prob.AddCols(1)
				prob.AddRows(1)
				prob.SetObjCoef(prob.NumCols(), inst.Conflicts.At(i, j))

				rowMask := make([]float64, prob.NumCols()+1)
				colMask := make([]float64, prob.NumRows()+1)
				rowMask[i+1] = 1
				rowMask[j+1] = 1
				rowMask[prob.NumCols()] = -1

				prob.SetRowBnds(prob.NumRows(), glpk.UP, 1, 1)
				prob.SetMatRow(prob.NumRows(), arange(0, prob.NumCols()+1), rowMask)
				prob.SetMatCol(prob.NumCols(), arange(0, prob.NumRows()+1), colMask)
			}
		}
	}
}

func (inst *SCPCSInstance) defSCPCS() *glpk.Prob {
	prob := glpk.New()
	inst.defBaseSCP(prob)
	inst.defConflicts(prob)
	return prob
}

func (inst *SCPCSInstance) Solve() (*SCPCSSolution, error) {
	prob := inst.defSCPCS()
	return inst.runMIPSolver(prob)
}
