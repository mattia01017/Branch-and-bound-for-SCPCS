package scpcs

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"gopkg.in/dnaeon/go-priorityqueue.v1"
)

func (inst *Instance) greedyRepair(node *Node) (*Solution, error) {
	pq := priorityqueue.New[int, float64](priorityqueue.MinHeap)
	selectedSubset := mat.NewVecDense(inst.NumSubsets, nil)
	selectedSubset.CloneFromVec(node.PrimalSolution.Subsets)

	for i := node.FixedSubsets; i < inst.NumSubsets; i++ {
		pq.Put(
			i,
			(inst.Costs.At(i, 0)+mat.Dot(selectedSubset, inst.Conflicts.RowView(i)))/mat.Sum(inst.Subsets.ColView(i)),
		)
	}

	for !inst.isFeasible(selectedSubset) {
		if pq.Len() == 0 {
			return nil, fmt.Errorf("Infeasible")
		}

		item := pq.Get()
		selectedSubset.SetVec(item.Value, 1)

		for i := range inst.NumSubsets {
			if inst.Conflicts.At(i, item.Value) > 0 {
				newCost := item.Priority + inst.Conflicts.At(i, item.Value)
				if node.PrimalSolution.Subsets.At(i, 0) < 0.5 {
					pq.Update(i, newCost/mat.Sum(inst.Subsets.ColView(i)))
				}
			}
		}
	}

	return &Solution{
		Subsets:   selectedSubset,
		TotalCost: inst.getCost(selectedSubset),
	}, nil
}
