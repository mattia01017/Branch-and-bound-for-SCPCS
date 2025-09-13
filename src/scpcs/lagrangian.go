package main

import (
	"fmt"
	"math"

	"github.com/lukpank/go-glpk/glpk"
	"gonum.org/v1/gonum/mat"
	"gopkg.in/dnaeon/go-priorityqueue.v1"
)

func (inst *SCPCSInstance) optimizeLagrangianDual(partialSol *SCPCSPartialSolution, lambda *mat.VecDense) (*SCPCSSolution, error) {
	prob := glpk.New()
	prob.SetObjDir(glpk.MIN)
	prob.AddCols(inst.NumSubsets)

	c := mat.NewVecDense(inst.NumSubsets, nil)
	prod := mat.NewVecDense(inst.NumSubsets, nil)
	prod.MulVec(inst.Subsets.T(), lambda)
	c.AddVec(inst.Costs, prod)

	for j := range inst.NumSubsets {
		prob.SetColKind(j+1, glpk.BV)
		prob.SetObjCoef(j+1, c.At(j, 0))
		if partialSol.FixedSubsets.At(j, 0) > 0.5 {
			prob.SetColBnds(j+1, glpk.FX, 1, 1)
		}
	}

	inst.defConflicts(prob)
	sol, err := inst.runMIPSolver(prob)
	if err != nil {
		return nil, err
	}
	b := mat.NewVecDense(inst.NumElements, nil)
	for i := range inst.NumElements {
		b.SetVec(i, 1)
	}
	sol.TotalCost += mat.Dot(lambda, b)
	return sol, nil
}

func (inst *SCPCSInstance) checkFeasibility(selectedMap map[int]bool) bool {
	for j := range inst.NumElements {
		for i, _ := range selectedMap {
			if inst.Subsets.At(i, j) == 1 {
				continue
			}
			return false
		}
	}
	return true
}

func (inst *SCPCSInstance) simpleGreedy(partialSol *SCPCSPartialSolution) (*SCPCSSolution, error) {
	selectedMap := make(map[int]bool)
	costs := mat.NewVecDense(inst.NumSubsets, nil)
	totalCost := partialSol.TotalCost
	pq := priorityqueue.New[int, float64](priorityqueue.MinHeap)

	for subset, cost := range inst.Costs.RawVector().Data {
		if partialSol.FixedSubsets.At(subset, 0) == 0 {
			selectedMap[subset] = true
			pq.Put(subset, cost)
		}
	}

	for !inst.checkFeasibility(selectedMap) {
		if pq.Len() == 0 {
			return nil, fmt.Errorf("INFEASIBLE")
		}

		item := pq.Get()

		selectedMap[item.Value] = true
		totalCost += item.Priority

		for i := range inst.Conflicts.RawRowView(item.Value) {
			if inst.Conflicts.At(i, item.Value) > 0.5 {
				newCost := costs.At(i, 0) + inst.Conflicts.At(i, item.Value)

				costs.SetVec(i, newCost)
				pq.Update(i, newCost)
			}
		}
	}

	sol := SCPCSSolution{
		SelectedSubsets: mat.NewVecDense(inst.NumSubsets, nil),
		TotalCost:       totalCost,
	}

	for i := range selectedMap {
		sol.SelectedSubsets.SetVec(i, 1)
	}

	return &sol, nil
}

func (inst *SCPCSInstance) subgradientOptimization(partialSol *SCPCSPartialSolution) (*SCPCSSolution, error) {
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
		sol, err = inst.optimizeLagrangianDual(partialSol, lambda)
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

		fmt.Println("L", lambda)
		fmt.Println("STEP", sol)

		for i := range inst.NumElements {
			lambda.SetVec(i, math.Max(0, lambda.At(i, 0)-step*violation.At(i, 0)))
		}

		lastCurrDiff = lastOptValue - sol.TotalCost
		lastOptValue = sol.TotalCost
		step *= rho
	}
	return sol, nil
}

func (inst *SCPCSInstance) SolveWithLagrangianRelaxation() (*SCPCSSolution, error) {
	nodesDeque := NewQueue[*SCPCSPartialSolution]()
	nodesDeque.Push(&SCPCSPartialSolution{
		TotalCost:       0,
		SelectedSubsets: mat.NewVecDense(inst.NumSubsets, nil),
	})

	primalBound := math.Inf(1)
	dualBound := -math.Inf(1)

	for nodesDeque.Size() > 0 {
		node := nodesDeque.Pop()

		sol, err := inst.simpleGreedy(node)
		if err.Error() == "INFEASIBLE" {
			continue
		}
		if sol.TotalCost < primalBound {
			primalBound = sol.TotalCost
		}

		sol, err = inst.subgradientOptimization(node)
		if err != nil {
			return nil, err
		}

		if sol.TotalCost > dualBound {
			dualBound = sol.TotalCost
		}
	}

	return nil, nil
}
