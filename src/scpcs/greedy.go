package scpcs

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"gopkg.in/dnaeon/go-priorityqueue.v1"
)

func (inst *Instance) greedyRepair(node *Node) (*Solution, error) {
	costs := mat.NewVecDense(inst.NumSubsets, nil)
	costs.CloneFromVec(inst.Costs)
	pq := priorityqueue.New[int, float64](priorityqueue.MinHeap)
	selectedSubset := mat.NewVecDense(inst.NumSubsets, nil)
	selectedSubset.CloneFromVec(node.PrimalSolution.Subsets)

	for i := range node.FixedSubsets {
		cost := inst.Costs.At(i, 0)
		if node.PrimalSolution.Subsets.At(i, 0) > 0.5 {
			pq.Put(i, float64(-i))
			cost += mat.Dot(selectedSubset, inst.Conflicts.RowView(i))
		}
		costs.SetVec(i, cost)
	}

	for i := node.FixedSubsets; i < inst.NumSubsets; i++ {
		costs.SetVec(i, inst.Costs.At(i, 0))
		if node.PrimalSolution.Subsets.At(i, 0) > 0.5 {
			pq.Put(i, float64(-i))
		} else {
			pq.Put(i, inst.Costs.At(i, 0)/mat.Sum(inst.Subsets.ColView(i)))
		}
	}

	for !inst.isFeasible(selectedSubset) {
		if pq.Len() == 0 {
			return nil, fmt.Errorf("Infeasible")
		}

		item := pq.Get()
		selectedSubset.SetVec(item.Value, 1)

		for i := range inst.NumSubsets {
			if inst.Conflicts.At(i, item.Value) > 0 {
				newCost := costs.At(i, 0) + inst.Conflicts.At(i, item.Value)/2
				costs.SetVec(i, newCost)
				if node.PrimalSolution.Subsets.At(i, 0) < 0.5 {
					pq.Update(i, newCost/mat.Sum(inst.Subsets.ColView(i)))
				}
			}
		}
	}

	return &Solution{
		Subsets:   selectedSubset,
		TotalCost: mat.Dot(selectedSubset, costs),
	}, nil
}
