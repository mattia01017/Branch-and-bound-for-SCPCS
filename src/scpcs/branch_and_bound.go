package scpcs

import (
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
)

const (
	eps = 1e-8

	bbTreeChildren   = 6
	subgradBaseStep  = 10.0
	subgradCoeffStep = 0.6
)

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
	return math.Abs(a-b) < eps
}

func (inst *Instance) fixSubsetInPartialSol(partialSol *Node, include []bool) *Node {
	newPartialSol := &Node{
		FixedSubsets: partialSol.FixedSubsets + len(include),
		PrimalSolution: &Solution{
			Subsets: mat.NewVecDense(
				inst.NumSubsets,
				slices.Clone(partialSol.PrimalSolution.Subsets.RawVector().Data),
			),
			TotalCost: partialSol.PrimalSolution.TotalCost,
		},
	}
	for i, flag := range include {
		if flag {
			newPartialSol.PrimalSolution.Subsets.SetVec(partialSol.FixedSubsets+i, 1)
			newPartialSol.PrimalSolution.TotalCost += inst.Costs.At(partialSol.FixedSubsets+i, 0) +
				mat.Dot(newPartialSol.PrimalSolution.Subsets, inst.Conflicts.ColView(partialSol.FixedSubsets+i))
		}
	}
	return newPartialSol
}

func (inst *Instance) SolveWithLagrangeanRelaxation() (*Solution, error) {
	lp := inst.defLagrangeanRelaxation()
	initialNode := &Node{
		PrimalSolution: &Solution{
			Subsets: mat.NewVecDense(inst.NumSubsets, nil),
		},
	}

	t := time.Now()
	bestPrimalSolution := inst.geneticHeuristic(initialNode, 1000)
	fmt.Println("Genetic algorithm time:", time.Since(t))
	fmt.Println("Genetic algorithm primal bound:", bestPrimalSolution.TotalCost)

	initialLB, lambda, err := inst.optimizeSubgradient(lp, initialNode)
	if err != nil {
		return nil, err
	}
	if almostEqual(0, (bestPrimalSolution.TotalCost-initialLB.TotalCost)/bestPrimalSolution.TotalCost) {
		return bestPrimalSolution, nil
	}
	initialNode.DualBound = initialLB.TotalCost
	initialNode.LagrangeanMul = lambda

	nodesDeque := NewStack[*Node]()
	nodesDeque.Push(initialNode)

	for nodesDeque.Size() > 0 {
		node := nodesDeque.Pop()
		fmt.Println(node)
		fmt.Println("Current UB:", bestPrimalSolution.TotalCost)

		if node.DualBound > bestPrimalSolution.TotalCost {
			continue
		}
		if node.PrimalSolution.TotalCost < bestPrimalSolution.TotalCost && inst.isFeasible(node.PrimalSolution.Subsets) {
			bestPrimalSolution = node.PrimalSolution
			continue
		}
		if node.FixedSubsets == inst.NumSubsets {
			continue
		}

		children := generateChildren(inst, node)

		errorCh := make(chan error)
		primalCh := make(chan *Solution)
		nodesCh := make(chan *Node, len(children))

		for _, n := range children {
			go func() {
				dualSol, lambda, err := inst.optimizeSubgradient(cloneLp(lp), n)
				if err != nil {
					errorCh <- err
					return
				}

				n.DualBound = dualSol.TotalCost
				n.LagrangeanMul = lambda

				repairedSol, err := inst.greedyRepair(n)
				if err != nil {
					errorCh <- err
					return
				}

				if n.DualBound <= bestPrimalSolution.TotalCost {
					nodesCh <- n
				}

				primalCh <- repairedSol
			}()
		}

		for range children {
			select {
			case err := <-errorCh:
				if err.Error() != "Infeasible" {
					return nil, err
				}
			case repairedSol := <-primalCh:
				if repairedSol.TotalCost < bestPrimalSolution.TotalCost {
					bestPrimalSolution = repairedSol
				}
			}
		}

		close(nodesCh)

		toPush := make([]*Node, 0, len(children))
		for n := range nodesCh {
			toPush = append(toPush, n)
		}

		slices.SortFunc(
			toPush,
			func(x, y *Node) int {
				if x.DualBound < y.DualBound {
					return 1
				}
				return -1
			},
		)

		for _, n := range toPush {
			nodesDeque.Push(n)
		}
	}

	return bestPrimalSolution, nil
}

func generateChildren(inst *Instance, node *Node) []*Node {
	treeChildren := int(math.Min(bbTreeChildren, float64(inst.NumSubsets-node.FixedSubsets+1)))
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
	return nodes
}
