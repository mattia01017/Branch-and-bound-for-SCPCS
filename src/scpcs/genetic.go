package scpcs

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/tomcraven/goga"
	"gonum.org/v1/gonum/mat"
)

const maximumRounds = 5000

type selectionSimulator struct {
	ElapsedRounds int
	MaximumRounds int
	Instance      *Instance
}

func getSelectedFromGenome(g goga.Genome) (selected *mat.VecDense) {
	bits := g.GetBits().GetAll()
	selected = mat.NewVecDense(len(bits), nil)
	for i, v := range bits {
		selected.SetVec(i, float64(v))
	}
	return
}

func (inst *Instance) getCost(selected *mat.VecDense) (cost float64) {
	cost = mat.Dot(selected, inst.Costs)
	for i := range inst.NumSubsets {
		for j := i + 1; j < inst.NumSubsets; j++ {
			if selected.At(i, 0) > 0.5 && selected.At(j, 0) > 0.5 {
				cost += inst.Conflicts.At(i, j)
			}
		}
	}
	return
}

func (bestGenome *selectionSimulator) OnBeginSimulation() {
}
func (sms *selectionSimulator) OnEndSimulation() {
	sms.ElapsedRounds++
}

func (sms *selectionSimulator) Simulate(g goga.Genome) {
	bits := g.GetBits().GetAll()
	selected := mat.NewVecDense(sms.Instance.NumSubsets, nil)
	for i, v := range bits {
		selected.SetVec(i, float64(v))
	}

	if sms.Instance.isFeasible(selected) {
		g.SetFitness(-int(sms.Instance.getCost(selected)))
	} else {
		g.SetFitness(math.MinInt)
	}
}
func (sms *selectionSimulator) ExitFunc(g goga.Genome) bool {
	return true
}

type myBitsetCreate struct {
	SolutionNode *Node
	Instance     *Instance
}

func (bc *myBitsetCreate) Go() goga.Bitset {
	b := goga.Bitset{}
	b.Create(bc.Instance.NumSubsets)
	for i := range bc.SolutionNode.FixedSubsets {
		b.Set(i, int(math.Round(bc.SolutionNode.PrimalSolution.Subsets.At(i, 0))))
	}
	for i := bc.SolutionNode.FixedSubsets; i < bc.Instance.NumSubsets; i++ {
		b.Set(i, rand.Intn(2))
	}
	return b
}

type myEliteConsumer struct {
	BestGenome goga.Genome
	Instance   *Instance
}

func (ec *myEliteConsumer) OnElite(g goga.Genome) {
	if (ec.BestGenome == nil || ec.BestGenome.GetFitness() < g.GetFitness()) && ec.Instance.isFeasible(getSelectedFromGenome(g)) {
		ec.BestGenome = g
	}
}

const populationSize = 1000

func (inst *Instance) geneticHeuristic(partialSol *Node, rounds int) *Solution {
	partialMutate := func(g1, g2 goga.Genome) (goga.Genome, goga.Genome) {
		g1BitsOrig := g1.GetBits()
		g1Bits := g1BitsOrig.CreateCopy()
		randomBit := partialSol.FixedSubsets + rand.Intn(inst.NumElements-partialSol.FixedSubsets)
		g1Bits.Set(randomBit, 1-g1Bits.Get(randomBit))
		return goga.NewGenome(g1Bits), goga.NewGenome(*g2.GetBits())
	}

	genAlgo := goga.NewGeneticAlgorithm()
	simulator := &selectionSimulator{
		MaximumRounds: maximumRounds,
		Instance:      inst,
	}
	genAlgo.Simulator = simulator
	genAlgo.BitsetCreate = &myBitsetCreate{
		Instance:     inst,
		SolutionNode: partialSol,
	}
	eliteConsumer := &myEliteConsumer{
		Instance: inst,
	}
	genAlgo.EliteConsumer = eliteConsumer
	genAlgo.Mater = goga.NewMater(
		[]goga.MaterFunctionProbability{
			{P: 0.9, F: goga.TwoPointCrossover, UseElite: true},
			{P: 0.9, F: goga.TwoPointCrossover},
			{P: 0.9, F: goga.TwoPointCrossover},
			{P: 0.9, F: partialMutate},
			{P: 0.9, F: partialMutate},
			{P: 0.9, F: partialMutate},
			{P: 0.9, F: partialMutate},
			{P: 0.9, F: partialMutate},
			{P: 0.9, F: partialMutate},
			{P: 0.9, F: goga.UniformCrossover},
		},
	)
	genAlgo.Selector = goga.NewSelector(
		[]goga.SelectorFunctionProbability{
			{P: 0.9, F: goga.Roulette},
		},
	)
	genAlgo.Init(populationSize, runtime.NumCPU())

	noImprovRounds := 0
	lastFitness := math.MinInt
	t := time.Now()
	genAlgo.SimulateUntil(func(g goga.Genome) bool {
		if g.GetFitness() == math.MinInt {
			return false
		}
		if g.GetFitness() == lastFitness {
			noImprovRounds++
		} else {
			noImprovRounds = 0
			lastFitness = g.GetFitness()
		}

		return noImprovRounds == rounds
	})
	fmt.Println("Genetic algorithm time:", time.Since(t))

	if eliteConsumer.BestGenome == nil {
		return &Solution{
			Subsets:   mat.NewVecDense(inst.NumSubsets, nil),
			TotalCost: math.Inf(1),
		}
	}

	selected := getSelectedFromGenome(eliteConsumer.BestGenome)

	return &Solution{
		Subsets:   selected,
		TotalCost: inst.getCost(selected),
	}
}
