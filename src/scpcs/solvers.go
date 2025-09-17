package main

import (
	"fmt"
	"math"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
)

func (inst *SCPCSInstance) runMIPSolver(lp *highs.Model) (*SCPCSSolution, error) {
	solution, err := lp.Solve()
	if err != nil {
		return nil, err
	}
	if solution.Status != highs.Optimal {
		return nil, fmt.Errorf("%v", solution.Status.String())
	}

	return &SCPCSSolution{
		SelectedSubsets: mat.NewVecDense(inst.NumSubsets, solution.ColumnPrimal[:inst.NumSubsets]),
		TotalCost:       solution.Objective,
	}, nil
}

func (inst *SCPCSInstance) defBaseSCP(lp *highs.Model) {
	numCols := inst.NumSubsets + len(inst.ConflictsList)
	infinity := math.Inf(1)

	lp.VarTypes = make([]highs.VariableType, numCols)
	lp.ColLower = make([]float64, numCols)
	lp.ColUpper = make([]float64, numCols)

	for j := range numCols {
		lp.VarTypes[j] = highs.IntegerType
		lp.ColUpper[j] = 1
	}

	row := make([]float64, numCols)
	copy(row, inst.Costs.RawVector().Data)
	for i, pair := range inst.ConflictsList {
		row[inst.NumSubsets+i] = inst.Conflicts.At(pair[0], pair[1])
	}

	lp.ColCosts = row

	for i := range inst.NumElements {
		lp.AddDenseRow(1, inst.Subsets.RawRowView(i), infinity)
	}
}

func (inst *SCPCSInstance) defConflicts(lp *highs.Model, insertedRows int) {
	for i, conflict := range inst.ConflictsList {
		lp.ConstMatrix = append(
			lp.ConstMatrix,
			highs.Nonzero{Row: insertedRows + i, Col: conflict[0], Val: 1},
			highs.Nonzero{Row: insertedRows + i, Col: conflict[1], Val: 1},
			highs.Nonzero{Row: insertedRows + i, Col: inst.NumSubsets + i, Val: -1},
		)
	}

	ub := make([]float64, len(inst.ConflictsList))
	lb := make([]float64, len(inst.ConflictsList))
	for i := range ub {
		ub[i] = 1
	}

	lp.RowUpper = append(lp.RowUpper, ub...)
	lp.RowLower = append(lp.RowLower, lb...)
}

func (inst *SCPCSInstance) defSCPCS() *highs.Model {
	lp := new(highs.Model)
	inst.defBaseSCP(lp)
	inst.defConflicts(lp, inst.NumElements)
	return lp
}

func (inst *SCPCSInstance) Solve() (*SCPCSSolution, error) {
	lp := inst.defSCPCS()
	return inst.runMIPSolver(lp)
}
