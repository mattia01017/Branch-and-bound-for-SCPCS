package scpcs

import (
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
)

const (
	subgradBaseStep  = 10.0
	subgradCoeffStep = 0.6
)

func (inst *Instance) optimizeSubgradient(lp *highs.Model, partialSol *Node) (sol *Solution, lambda *mat.VecDense, err error) {
	lambda = mat.NewVecDense(inst.NumElements, nil)
	if partialSol.LagrangeanMul == nil {
		for i := range inst.NumElements {
			lambda.SetVec(i, 1)
		}
	} else {
		lambda.CloneFromVec(partialSol.LagrangeanMul)
	}

	best := &Solution{TotalCost: math.Inf(-1)}

	noImprovementRounds := 0
	step := subgradBaseStep

	for {
		sol, err = inst.solveLagrangeanPrimal(lp, partialSol, lambda)
		if math.IsNaN(sol.TotalCost) {
			fmt.Fprintln(os.Stderr, "Obtained NaN in subgradient method")
			return
		}
		if err != nil {
			return
		}

		improvement := sol.TotalCost - best.TotalCost
		if improvement > 0 {
			best = sol
		}
		if improvement < 0.1 {
			if noImprovementRounds == 5 {
				return
			}
			noImprovementRounds++
		} else {
			noImprovementRounds = 0
		}
		step *= subgradCoeffStep

		Ax := mat.NewVecDense(inst.NumElements, nil)
		Ax.MulVec(inst.Subsets, sol.Subsets)

		violations := mat.NewVecDense(inst.NumElements, nil)
		for j := range inst.NumElements {
			violations.SetVec(j, 1.0-Ax.At(j, 0))
		}

		for j := range lambda.Len() {
			lambda.SetVec(j, math.Max(0, lambda.At(j, 0)+step*violations.At(j, 0)))
		}
	}
}

func (inst *Instance) defLagrangeanRelaxation() *highs.Model {
	numCols := inst.NumSubsets + len(inst.ConflictsList)
	lp := new(highs.Model)

	lp.ConstMatrix = make([]highs.Nonzero, 1)
	lp.VarTypes = make([]highs.VariableType, numCols)
	for j := range inst.NumSubsets {
		lp.VarTypes[j] = highs.IntegerType
	}

	inst.defConflicts(lp, 0)
	return lp
}

func (inst *Instance) solveLagrangeanPrimal(lp *highs.Model, partialSol *Node, lambda *mat.VecDense) (*Solution, error) {
	lp.ColCosts = inst.getLagrangeanCosts(lambda)
	lp.ColLower = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	lp.ColUpper = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	for j := range partialSol.FixedSubsets {
		lp.ColLower[j] = partialSol.PrimalSolution.Subsets.At(j, 0)
		lp.ColUpper[j] = partialSol.PrimalSolution.Subsets.At(j, 0)
	}
	for j := partialSol.FixedSubsets; j < len(lp.ColLower); j++ {
		lp.ColLower[j] = 0
		lp.ColUpper[j] = 1
	}

	lp.Offset = mat.Sum(lambda)
	sol, err := inst.runHighsSolver(lp)
	if err != nil {
		return nil, err
	}

	return sol, nil
}

func (inst *Instance) getLagrangeanCosts(lambda *mat.VecDense) []float64 {
	c := mat.NewVecDense(inst.NumSubsets, nil)
	prod := mat.NewVecDense(inst.NumSubsets, nil)
	prod.MulVec(inst.Subsets.T(), lambda)
	c.SubVec(inst.Costs, prod)

	conflictCosts := make([]float64, len(inst.ConflictsList))
	for i, pair := range inst.ConflictsList {
		conflictCosts[i] = inst.Conflicts.At(pair[0], pair[1])
	}

	objCoeff := slices.Clone(c.RawVector().Data)
	objCoeff = append(objCoeff, conflictCosts...)
	return objCoeff
}

func (inst *Instance) isFeasible(selected *mat.VecDense) bool {
	Ax := mat.NewVecDense(inst.NumElements, nil)
	Ax.MulVec(inst.Subsets, selected)

	for i := range inst.NumElements {
		if almostEqual(Ax.At(i, 0), 0) {
			return false
		}
	}
	return true
}
