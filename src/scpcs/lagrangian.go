package main

import (
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
	"gopkg.in/dnaeon/go-priorityqueue.v1"
)

const eps = 1e-8

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < eps
}

func (inst *SCPCSInstance) defLagrangianRelaxation() *highs.Model {
	numCols := inst.NumSubsets + len(inst.ConflictsList)
	lp := new(highs.Model)

	lp.VarTypes = make([]highs.VariableType, numCols)
	for j := range numCols {
		lp.VarTypes[j] = highs.IntegerType
	}

	inst.defConflicts(lp, 0)
	return lp
}

func (inst *SCPCSInstance) checkLagrangianOptimality(sol *SCPCSSolution, lambda *mat.VecDense) bool {
	Ax := mat.NewVecDense(inst.NumElements, nil)
	Ax.MulVec(inst.Subsets, sol.SelectedSubsets)
	for i := range inst.NumElements {
		if Ax.At(i, 0) < 1 || !almostEqual(0, lambda.At(i, 0)*(1.0-Ax.At(i, 0))) {
			return false
		}
	}
	return true
}

func (inst *SCPCSInstance) optimizeLagrangianPrimal(lp *highs.Model, partialSol *SCPCSPartialSolution, lambda *mat.VecDense) (*SCPCSSolution, error) {
	lp.ColCosts = inst.getLagrangianCosts(lambda)
	lp.ColLower = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	lp.ColUpper = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	for j := range partialSol.FixedSubsets {
		lp.ColLower[j] = partialSol.SelectedSubsets.At(j, 0)
		lp.ColUpper[j] = partialSol.SelectedSubsets.At(j, 0)
	}
	for j := partialSol.FixedSubsets; j < len(lp.ColLower); j++ {
		lp.ColLower[j] = 0
		lp.ColUpper[j] = 1
	}

	lp.Offset = mat.Sum(lambda)
	sol, err := inst.runMIPSolver(lp)
	if err != nil {
		return nil, err
	}

	return sol, nil
}

func (inst *SCPCSInstance) getLagrangianCosts(lambda *mat.VecDense) []float64 {
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
	costs.CloneFromVec(inst.Costs)
	pq := priorityqueue.New[int, float64](priorityqueue.MinHeap)

	for i := range inst.NumSubsets {
		costs.SetVec(i, inst.Costs.At(i, 0))
		if initialSol.SelectedSubsets.At(i, 0) > 0.5 {
			pq.Put(i, -1)
		} else {
			pq.Put(i, inst.Costs.At(i, 0))
		}
	}

	for !inst.checkFeasibility(selectedMap) {
		if pq.Len() == 0 {
			return nil, fmt.Errorf("Infeasible")
		}

		item := pq.Get()
		selectedMap[item.Value] = true

		for i := range inst.NumSubsets {
			if inst.Conflicts.At(i, item.Value) > 0 {
				newCost := costs.At(i, 0) + inst.Conflicts.At(i, item.Value)
				costs.SetVec(i, newCost)
				pq.Update(i, newCost)
			}
		}
	}

	selectedSubset := mat.NewVecDense(inst.NumSubsets, nil)

	totalCost := 0.0
	for i := range selectedMap {
		selectedSubset.SetVec(i, 1)
		totalCost += costs.At(i, 0)
	}

	return &SCPCSSolution{
		SelectedSubsets: selectedSubset,
		TotalCost:       totalCost,
	}, nil
}

func (inst *SCPCSInstance) optimizeSubgradient(lp *highs.Model, partialSol *SCPCSPartialSolution, ub float64) (*SCPCSSolution, bool, error) {
	lambda := mat.NewVecDense(inst.NumElements, nil)
	pi := 2.0
	lastOptValue := math.Inf(1)
	var noImprovementRounds int
	var lastCurrDiff float64
	var sol *SCPCSSolution
	var err error

	for {
		sol, err = inst.optimizeLagrangianPrimal(lp, partialSol, lambda)
		if err != nil {
			return nil, false, err
		}

		Ax := mat.NewVecDense(inst.NumElements, nil)
		Ax.MulVec(inst.Subsets, sol.SelectedSubsets)

		sqSum := 0.0
		violations := mat.NewVecDense(inst.NumElements, nil)
		for j := range inst.NumElements {
			violations.SetVec(j, 1.0-Ax.At(j, 0))
			sqSum += math.Pow(math.Max(0, violations.At(j, 0)), 2)
		}

		if sqSum == 0 {
			sqSum = 1
		}

		lastCurrDiff = math.Abs(sol.TotalCost - lastOptValue)
		lastOptValue = sol.TotalCost

		if lastCurrDiff < 0.05 {
			if noImprovementRounds == 5 {
				break
			}
			noImprovementRounds++
		} else {
			noImprovementRounds = 0
		}

		step := pi * (ub - sol.TotalCost) / sqSum
		pi /= 2
		for j := range lambda.Len() {
			lambda.SetVec(j, math.Max(0, lambda.At(j, 0)+step*violations.At(j, 0)))
		}
	}
	return sol, inst.checkLagrangianOptimality(sol, lambda), nil
}

func (inst *SCPCSInstance) fixSubsetInPartialSol(partialSol *SCPCSPartialSolution, include bool) *SCPCSPartialSolution {
	newPartialSol := &SCPCSPartialSolution{
		FixedSubsets:    partialSol.FixedSubsets + 1,
		SelectedSubsets: mat.VecDenseCopyOf(partialSol.SelectedSubsets),
	}
	if include {
		newPartialSol.SelectedSubsets.SetVec(partialSol.FixedSubsets, 1)
	}
	return newPartialSol
}

func (inst *SCPCSInstance) SolveWithLagrangianRelaxation() (*SCPCSSolution, error) {
	emptyVec := mat.NewVecDense(inst.NumSubsets, nil)
	nodesDeque := priorityqueue.New[*SCPCSPartialSolution, float64](priorityqueue.MinHeap)
	nodesDeque.Put(&SCPCSPartialSolution{SelectedSubsets: emptyVec}, 0)

	lp := inst.defLagrangianRelaxation()

	var start time.Time
	firstSol, _ := inst.greedyRepair(&SCPCSSolution{SelectedSubsets: emptyVec})
	bestPrimalBound := firstSol.TotalCost
	for nodesDeque.Len() > 0 {
		node := nodesDeque.Get()

		start = time.Now()
		sol, optimal, err := inst.optimizeSubgradient(lp, node.Value, bestPrimalBound)
		fmt.Println(sol)
		fmt.Println("Subgradient opt:", time.Since(start))
		if optimal {
			return sol, nil
		}
		if err != nil {
			if err.Error() == "Infeasible" {
				continue
			}
			return nil, err
		}
		dualBound := sol.TotalCost
		if dualBound > bestPrimalBound {
			continue
		}

		sol, err = inst.greedyRepair(sol)
		if err != nil {
			return nil, err
		}
		if sol.TotalCost < bestPrimalBound {
			bestPrimalBound = sol.TotalCost
		}
		if sol.TotalCost <= dualBound {
			return sol, nil
		}

		if node.Value.FixedSubsets != inst.NumSubsets {
			nodesDeque.Put(inst.fixSubsetInPartialSol(node.Value, false), dualBound)
			nodesDeque.Put(inst.fixSubsetInPartialSol(node.Value, true), dualBound)
		}
	}

	return nil, fmt.Errorf("couldn't find optimal solution")
}
