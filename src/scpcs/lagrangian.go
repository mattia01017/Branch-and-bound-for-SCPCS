package scpcs

import (
	"fmt"
	"math"
	"slices"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
	"gopkg.in/dnaeon/go-priorityqueue.v1"
)

const EPS = 1e-8
const TREE_CHILDREN = 6

func cloneLp(lp *highs.Model) *highs.Model {
	return &highs.Model{
		Maximize:      lp.Maximize,
		ColCosts:      slices.Clone(lp.ColCosts),
		Offset:        lp.Offset,
		ColLower:      slices.Clone(lp.ColLower),
		ColUpper:      slices.Clone(lp.ColUpper),
		RowLower:      slices.Clone(lp.RowLower),
		RowUpper:      slices.Clone(lp.RowUpper),
		ConstMatrix:   slices.Clone(lp.ConstMatrix),
		HessianMatrix: slices.Clone(lp.HessianMatrix),
		VarTypes:      slices.Clone(lp.VarTypes),
	}
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < EPS
}

func (inst *Instance) defLagrangianRelaxation() *highs.Model {
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

func (inst *Instance) solveLagrangianPrimal(lp *highs.Model, partialSol *Node, lambda *mat.VecDense) (*Solution, error) {
	lp.ColCosts = inst.getLagrangianCosts(lambda)
	lp.ColLower = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	lp.ColUpper = make([]float64, inst.NumSubsets+len(inst.ConflictsList))
	for j := range partialSol.FixedSubsets {
		lp.ColLower[j] = partialSol.CurrentSolution.SelectedSubsets.At(j, 0)
		lp.ColUpper[j] = partialSol.CurrentSolution.SelectedSubsets.At(j, 0)
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

func (inst *Instance) getLagrangianCosts(lambda *mat.VecDense) []float64 {
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

func (inst *Instance) greedyRepair(node *Node) (*Solution, error) {
	costs := mat.NewVecDense(inst.NumSubsets, nil)
	costs.CloneFromVec(inst.Costs)
	pq := priorityqueue.New[int, float64](priorityqueue.MinHeap)
	selectedSubset := mat.NewVecDense(inst.NumSubsets, nil)
	selectedSubset.CloneFromVec(node.CurrentSolution.SelectedSubsets)

	for i := range node.FixedSubsets {
		costs.SetVec(i, inst.Costs.At(i, 0))
		if node.CurrentSolution.SelectedSubsets.At(i, 0) > 0.5 {
			pq.Put(i, float64(-i))
		}
	}

	for i := node.FixedSubsets; i < inst.NumSubsets; i++ {
		costs.SetVec(i, inst.Costs.At(i, 0))
		if node.CurrentSolution.SelectedSubsets.At(i, 0) > 0.5 {
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
				if node.CurrentSolution.SelectedSubsets.At(i, 0) < 0.5 {
					pq.Update(i, newCost/mat.Sum(inst.Subsets.ColView(i)))
				}
			}
		}
	}

	totalCost := 0.0
	for i := range inst.NumSubsets {
		if selectedSubset.At(i, 0) > 0.5 {
			totalCost += costs.At(i, 0)
		}
	}

	return &Solution{
		SelectedSubsets: selectedSubset,
		TotalCost:       totalCost,
	}, nil
}

func (inst *Instance) optimizeSubgradient(lp *highs.Model, partialSol *Node) (*Solution, error) {
	lambda := mat.NewVecDense(inst.NumElements, nil)
	best := &Solution{TotalCost: math.Inf(-1)}
	var sol *Solution
	var err error

	noImprovementRounds := 0
	step := 10.0
	mult := 0.6

	for {
		sol, err = inst.solveLagrangianPrimal(lp, partialSol, lambda)
		if err != nil {
			return nil, err
		}
		diff := best.TotalCost - sol.TotalCost
		if diff < 0 {
			best = sol
		}
		if diff < 1e-2 {
			noImprovementRounds++
			step *= mult
		} else {
			noImprovementRounds = 0
		}

		if noImprovementRounds == 10 {
			return best, nil
		}

		Ax := mat.NewVecDense(inst.NumElements, nil)
		Ax.MulVec(inst.Subsets, sol.SelectedSubsets)

		violations := mat.NewVecDense(inst.NumElements, nil)
		for j := range inst.NumElements {
			violations.SetVec(j, 1.0-Ax.At(j, 0))
		}

		for j := range lambda.Len() {
			lambda.SetVec(j, math.Max(0, lambda.At(j, 0)+step*violations.At(j, 0)))
		}

	}
}

func (inst *Instance) fixSubsetInPartialSol(partialSol *Node, include []bool) *Node {
	newPartialSol := &Node{
		FixedSubsets: partialSol.FixedSubsets + len(include),
		CurrentSolution: &Solution{
			SelectedSubsets: mat.NewVecDense(
				inst.NumSubsets,
				slices.Clone(partialSol.CurrentSolution.SelectedSubsets.RawVector().Data),
			),
			TotalCost: partialSol.CurrentSolution.TotalCost,
		},
	}
	for i, flag := range include {
		if flag {
			newPartialSol.CurrentSolution.SelectedSubsets.SetVec(partialSol.FixedSubsets+i, 1)
			newPartialSol.CurrentSolution.TotalCost += inst.Costs.At(partialSol.FixedSubsets+i, 0) +
				mat.Dot(newPartialSol.CurrentSolution.SelectedSubsets, inst.Conflicts.ColView(partialSol.FixedSubsets+i))
		}
	}
	return newPartialSol
}

func (inst *Instance) SolveWithLagrangianRelaxation() (*Solution, error) {
	lp := inst.defLagrangianRelaxation()
	initialNode := &Node{
		CurrentSolution: &Solution{
			SelectedSubsets: mat.NewVecDense(inst.NumSubsets, nil),
		},
	}

	primalSol, _ := inst.greedyRepair(initialNode)
	initialLB, err := inst.optimizeSubgradient(lp, initialNode)
	if err != nil {
		return nil, err
	}
	if almostEqual(0, (primalSol.TotalCost-initialLB.TotalCost)/primalSol.TotalCost) {
		return primalSol, nil
	}
	initialNode.DualBound = initialLB.TotalCost

	nodesDeque := priorityqueue.New[*Node, float64](priorityqueue.MaxHeap)
	nodesDeque.Put(initialNode, 0)

	for nodesDeque.Len() > 0 {
		node := nodesDeque.Get().Value
		fmt.Println(node)
		fmt.Println("Current UB:", primalSol.TotalCost)

		if node.DualBound >= primalSol.TotalCost {
			continue
		}

		if node.CurrentSolution.TotalCost < primalSol.TotalCost && inst.isFeasible(node.CurrentSolution.SelectedSubsets) {
			primalSol = node.CurrentSolution
		} else if node.FixedSubsets < inst.NumSubsets {
			treeChildren := int(math.Min(TREE_CHILDREN, float64(inst.NumSubsets-node.FixedSubsets+1)))
			nodes := make([]*Node, 0, treeChildren)

			for i := range treeChildren - 2 {
				flags := make([]bool, i+1)
				flags[i] = true
				nodes = append(nodes, inst.fixSubsetInPartialSol(node, flags))
			}
			flags := make([]bool, treeChildren-1)
			nodes = append(nodes, inst.fixSubsetInPartialSol(node, flags))
			flags[len(flags)-1] = true
			nodes = append(nodes, inst.fixSubsetInPartialSol(node, flags))

			errorCh := make(chan error)
			primalCh := make(chan *Solution)
			optimalCh := make(chan *Solution)

			for _, n := range nodes {
				go func() {
					dualSol, err := inst.optimizeSubgradient(cloneLp(lp), n)
					if err != nil {
						errorCh <- err
						return
					}

					n.DualBound = dualSol.TotalCost

					repairedSol, err := inst.greedyRepair(n)
					if err != nil {
						errorCh <- err
						return
					}

					if n.DualBound < primalSol.TotalCost {
						nodesDeque.Put(n, n.DualBound)
					}

					ub := math.Min(repairedSol.TotalCost, primalSol.TotalCost)
					if almostEqual(0, (ub-n.DualBound)/ub) {
						optimalCh <- repairedSol
					} else {
						primalCh <- repairedSol
					}

				}()
			}

			for range nodes {
				select {
				case err := <-errorCh:
					if err.Error() != "Infeasible" {
						return nil, err
					}
				case repairedSol := <-primalCh:
					if repairedSol.TotalCost < primalSol.TotalCost {
						primalSol = repairedSol
					}
				case optimalSol := <-optimalCh:
					return optimalSol, nil
				}
			}
		}
	}

	return primalSol, nil
}
