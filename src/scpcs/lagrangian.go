package scpcs

import (
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/lanl/highs"
	"gonum.org/v1/gonum/mat"
	"gopkg.in/dnaeon/go-priorityqueue.v1"
)

const EPS = 1e-8
const TREE_CHILDREN = 6

const STEP_BASE = 10.0
const STEP_MUL = 0.6

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

func (inst *Instance) greedyRepair(node *Node) (*Solution, error) {
	costs := mat.NewVecDense(inst.NumSubsets, nil)
	costs.CloneFromVec(inst.Costs)
	pq := priorityqueue.New[int, float64](priorityqueue.MinHeap)
	selectedSubset := mat.NewVecDense(inst.NumSubsets, nil)
	selectedSubset.CloneFromVec(node.CurrentSolution.SelectedSubsets)

	for i := range node.FixedSubsets {
		cost := inst.Costs.At(i, 0)
		if node.CurrentSolution.SelectedSubsets.At(i, 0) > 0.5 {
			pq.Put(i, float64(-i))
			cost += mat.Dot(selectedSubset, inst.Conflicts.RowView(i))
		}
		costs.SetVec(i, cost)
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

	return &Solution{
		SelectedSubsets: selectedSubset,
		TotalCost:       mat.Dot(selectedSubset, costs),
	}, nil
}

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
	step := STEP_BASE

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
		step *= STEP_MUL

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

func (inst *Instance) SolveWithLagrangeanRelaxation() (*Solution, error) {
	lp := inst.defLagrangeanRelaxation()
	initialNode := &Node{
		CurrentSolution: &Solution{
			SelectedSubsets: mat.NewVecDense(inst.NumSubsets, nil),
		},
	}

	primalSol, _ := inst.greedyRepair(initialNode)
	initialLB, lambda, err := inst.optimizeSubgradient(lp, initialNode)
	if err != nil {
		return nil, err
	}
	if almostEqual(0, (primalSol.TotalCost-initialLB.TotalCost)/primalSol.TotalCost) {
		return primalSol, nil
	}
	initialNode.DualBound = initialLB.TotalCost
	initialNode.LagrangeanMul = lambda

	nodesDeque := NewStack[*Node]()
	nodesDeque.Push(initialNode)

	for nodesDeque.Size() > 0 {
		node := nodesDeque.Pop()
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
			nodesCh := make(chan *Node, treeChildren)

			for _, n := range nodes {
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

					if n.DualBound < primalSol.TotalCost {
						nodesCh <- n
					}

					ub := math.Min(repairedSol.TotalCost, primalSol.TotalCost)
					if almostEqual(0, (ub-n.DualBound)/ub) {
						// optimalCh <- repairedSol
						primalCh <- repairedSol
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

			close(nodesCh)

			toPush := make([]*Node, 0, treeChildren)
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
				fmt.Printf("\t%v\n", n)
				nodesDeque.Push(n)
			}
		}
	}

	return primalSol, nil
}
