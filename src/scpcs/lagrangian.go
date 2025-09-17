package main

import (
	"fmt"
	"math"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
	"gopkg.in/dnaeon/go-priorityqueue.v1"
)

func (inst *SCPCSInstance) defLagriangianRelaxation() *highs.Model {
	numCols := inst.NumSubsets + len(inst.ConflictsList)
	lp := new(highs.Model)

	integrality := make([]highs.VariableType, numCols)
	lb := make([]float64, numCols)
	ub := make([]float64, numCols)
	for j := range numCols {
		integrality[j] = highs.IntegerType
		ub[j] = 1
	}
	lp.ColLower = lb
	lp.ColUpper = ub
	lp.VarTypes = integrality

	inst.defConflicts(lp, 0)
	return lp
}

func (inst *SCPCSInstance) optimizeLagrangian(lp *highs.Model, partialSol *SCPCSPartialSolution, lambda *mat.VecDense) (*SCPCSSolution, error) {
	lp.ColCosts = inst.computeObjCoeff(lambda)
	lp.ColLower = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	lp.ColUpper = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	for j := range inst.NumSubsets {
		if j < partialSol.FixedSubsets {
			lp.ColLower[j] = partialSol.SelectedSubsets.At(j, 0)
			lp.ColUpper[j] = partialSol.SelectedSubsets.At(j, 0)
		} else {
			lp.ColLower[j] = 0
			lp.ColUpper[j] = 1
		}
	}

	lp.Offset = 0
	b := mat.NewVecDense(inst.NumElements, nil)
	for i := range inst.NumElements {
		b.SetVec(i, 1)
	}
	lp.Offset += mat.Dot(lambda, b)

	sol, err := inst.runMIPSolver(lp)
	if err != nil {
		return nil, err
	}
	return sol, nil
}

func (inst *SCPCSInstance) computeObjCoeff(lambda *mat.VecDense) []float64 {
	c := mat.NewVecDense(inst.NumSubsets, nil)
	prod := mat.NewVecDense(inst.NumSubsets, nil)
	prod.MulVec(inst.Subsets.T(), lambda)
	c.AddVec(inst.Costs, prod)

	conflictCosts := make([]float64, len(inst.ConflictsList))
	for i, conflict := range inst.ConflictsList {
		conflictCosts[i] = inst.Conflicts.At(conflict[0], conflict[1])
	}
	objCoeff := make([]float64, inst.NumSubsets+len(conflictCosts))
	copy(objCoeff, c.RawVector().Data)
	copy(objCoeff[inst.NumSubsets:], conflictCosts)
	return objCoeff
}

func (inst *SCPCSInstance) checkFeasibility(selectedMap map[int]bool) bool {
	for i := range inst.NumElements {
		found := false
		for j := range selectedMap {
			if inst.Subsets.At(i, j) > 0.5 {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (inst *SCPCSInstance) greedyRepair(initialSol *SCPCSSolution) (*SCPCSSolution, error) {
	selectedMap := make(map[int]bool)
	costs := mat.NewVecDense(inst.NumSubsets, nil)
	totalCost := initialSol.TotalCost
	pq := priorityqueue.New[int, float64](priorityqueue.MinHeap)

	for i := range inst.NumSubsets {
		if initialSol.SelectedSubsets.At(i, 0) > 0.5 {
			selectedMap[i] = true
		} else {
			pq.Put(i, inst.Costs.At(i, 0))
		}
	}

	for i := range inst.NumSubsets {
		if initialSol.SelectedSubsets.At(i, 0) > 0.5 {
			for j := range inst.NumSubsets {
				if initialSol.SelectedSubsets.At(i, 0) < 0.5 && inst.Conflicts.At(j, i) > 0 {
					cost := inst.Costs.At(j, 0) + inst.Conflicts.At(i, j)
					costs.SetVec(j, cost)
					pq.Update(j, cost)
				}
			}
		}
	}

	if inst.checkFeasibility(selectedMap) {
		return nil, fmt.Errorf("Optimal")
	}

	for !inst.checkFeasibility(selectedMap) {
		if pq.Len() == 0 {
			return nil, fmt.Errorf("Infeasible")
		}

		item := pq.Get()

		selectedMap[item.Value] = true
		totalCost += item.Priority

		for i := range inst.NumSubsets {
			if inst.Conflicts.At(i, item.Value) > 0 {
				newCost := costs.At(i, 0) + inst.Conflicts.At(i, item.Value)
				costs.SetVec(i, newCost)
				pq.Update(i, newCost)
			}
		}
	}

	selectedSubset := mat.NewVecDense(inst.NumSubsets, nil)

	sol := &SCPCSSolution{
		SelectedSubsets: selectedSubset,
		TotalCost:       totalCost,
	}

	for i := range selectedMap {
		sol.SelectedSubsets.SetVec(i, 1)
	}

	return sol, nil
}

func (inst *SCPCSInstance) subgradientOptimization(lp *highs.Model, partialSol *SCPCSPartialSolution) (*SCPCSSolution, error) {
	lambda := mat.NewVecDense(inst.NumElements, nil)
	for i := range inst.NumElements {
		lambda.SetVec(i, 1)
	}
	step := 1.0
	rho := 0.8

	lastOptValue := math.Inf(1)
	lastCurrDiff := math.Inf(1)
	var sol *SCPCSSolution
	var err error
	for lastCurrDiff > 1e-5 {
		sol, err = inst.optimizeLagrangian(lp, partialSol, lambda)
		if err != nil {
			return nil, err
		}

		violation := mat.NewVecDense(inst.NumElements, nil)
		for i := range inst.NumElements {
			violation.SetVec(i, 1)
		}

		Ax := mat.NewVecDense(inst.NumElements, nil)
		Ax.MulVec(inst.Subsets, sol.SelectedSubsets)
		violation.SubVec(violation, Ax)

		for i := range inst.NumElements {
			lambda.SetVec(i, math.Max(0, lambda.At(i, 0)-step*violation.At(i, 0)))
		}

		lastCurrDiff = lastOptValue - sol.TotalCost
		lastOptValue = sol.TotalCost
		step *= rho
	}
	return sol, nil
}

func (inst *SCPCSInstance) fixSubsetInPartialSol(partialSol *SCPCSPartialSolution, include bool) *SCPCSPartialSolution {
	newPartialSol := &SCPCSPartialSolution{
		FixedSubsets:    partialSol.FixedSubsets + 1,
		SelectedSubsets: mat.VecDenseCopyOf(partialSol.SelectedSubsets),
		TotalCost:       partialSol.TotalCost,
	}
	if include {
		newPartialSol.SelectedSubsets.SetVec(partialSol.FixedSubsets, 1)
		newPartialSol.TotalCost += mat.Dot(partialSol.SelectedSubsets, inst.Costs) + inst.Costs.At(partialSol.FixedSubsets, 0)
	}
	return newPartialSol
}

func (inst *SCPCSInstance) SolveWithLagrangianRelaxation() (*SCPCSSolution, error) {
	nodesDeque := NewStack[*SCPCSPartialSolution]()
	nodesDeque.Push(&SCPCSPartialSolution{
		SelectedSubsets: mat.NewVecDense(inst.NumSubsets, nil),
	})

	lp := inst.defLagriangianRelaxation()

	var sol *SCPCSSolution
	var err error
	primalBound := math.Inf(1)
	for nodesDeque.Size() > 0 {
		node := nodesDeque.Pop()

		sol, err = inst.subgradientOptimization(lp, node)
		if err != nil {
			if err.Error() == "Infeasible" {
				continue
			}
			return nil, err
		}
		if sol.TotalCost > primalBound {
			continue
		}

		sol, err = inst.greedyRepair(sol)
		if err != nil {
			if err.Error() == "Infeasible" {
				continue
			} else if err.Error() == "Optimal" {
				return sol, nil
			}
			return nil, err
		}
		if sol.TotalCost < primalBound {
			primalBound = sol.TotalCost
		}

		if node.FixedSubsets != inst.NumSubsets {
			nodesDeque.Push(inst.fixSubsetInPartialSol(node, false))
			nodesDeque.Push(inst.fixSubsetInPartialSol(node, true))
		}

	}

	return sol, nil
}
