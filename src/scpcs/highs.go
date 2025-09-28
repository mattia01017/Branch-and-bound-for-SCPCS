package scpcs

import (
	"fmt"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
)

func (inst *Instance) runHighsSolver(lp *highs.Model) (*Solution, error) {
	solution, err := lp.Solve()
	if err != nil {
		return nil, err
	}
	if solution.Status != highs.Optimal {
		return nil, fmt.Errorf("status: %v", solution.Status.String())
	}

	return &Solution{
		Subsets:   mat.NewVecDense(inst.NumSubsets, solution.ColumnPrimal[:inst.NumSubsets]),
		TotalCost: solution.Objective,
	}, nil
}

func (inst *Instance) defBaseSCP(lp *highs.Model) {
	numCols := inst.NumSubsets + len(inst.ConflictsList)

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
		lp.AddDenseRow(1, inst.Subsets.RawRowView(i), float64(inst.NumElements))
	}
}

func (inst *Instance) defConflicts(lp *highs.Model, rowsOffset int) {
	for i, pair := range inst.ConflictsList {
		lp.ConstMatrix = append(
			lp.ConstMatrix,
			highs.Nonzero{Row: rowsOffset + i, Col: pair[0], Val: 1},
			highs.Nonzero{Row: rowsOffset + i, Col: pair[1], Val: 1},
			highs.Nonzero{Row: rowsOffset + i, Col: inst.NumSubsets + i, Val: -1},
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

func (inst *Instance) defSCPCS() *highs.Model {
	lp := new(highs.Model)
	inst.defBaseSCP(lp)
	inst.defConflicts(lp, inst.NumElements)
	return lp
}

func (inst *Instance) Solve() (*Solution, error) {
	lp := inst.defSCPCS()
	return inst.runHighsSolver(lp)
}
