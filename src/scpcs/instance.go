package scpcs

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func errorCoalesce(args ...error) error {
	for _, e := range args {
		if e != nil {
			return e
		}
	}
	return nil
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

func (inst *Instance) parseFirstLine(scanner *bufio.Scanner) error {
	scanner.Scan()
	line := strings.Fields(scanner.Text())
	numElements, err := strconv.Atoi(line[0])
	if err != nil {
		return fmt.Errorf("error while parsing first line: %v", err)
	}
	numSubsets, err := strconv.Atoi(line[1])
	if err != nil {
		return fmt.Errorf("error while parsing first line: %v", err)
	}

	inst.NumElements = numElements
	inst.NumSubsets = numSubsets
	inst.Subsets = mat.NewDense(inst.NumElements, numSubsets, nil)
	inst.Costs = mat.NewVecDense(numSubsets, nil)

	return nil
}

func (inst *Instance) parseSecondLine(scanner *bufio.Scanner) error {
	scanner.Scan()
	line := strings.Fields(scanner.Text())
	for i, tok := range line {
		v, err := strconv.Atoi(tok)
		if err != nil {
			return fmt.Errorf("error while parsing second line: %v", err)
		}
		inst.Costs.SetVec(i, float64(v))
	}
	return nil
}

func (inst *Instance) parseIncompSets(scanner *bufio.Scanner) error {
	i := 0
	for scanner.Scan() {
		line := strings.Fields(scanner.Text())[1:]
		for _, tok := range line {
			v, err := strconv.Atoi(tok)
			if err != nil {
				return fmt.Errorf("error while parsing incompatibility set %d: %v", i, err)
			}
			inst.Subsets.Set(i, v-1, 1)
		}
		i++
	}
	return nil
}

func (inst *Instance) computeConflicts(conflictThreshold int) error {
	inst.Conflicts = mat.NewDense(inst.NumSubsets, inst.NumSubsets, nil)
	inst.ConflictsList = make([][]int, 0)
	coeffs := mat.NewVecDense(inst.NumSubsets, nil)
	for i := range inst.NumSubsets {
		coeffs.SetVec(i, inst.Costs.At(i, 0)/mat.Sum(inst.Subsets.ColView(i)))
	}
	coeff := math.Round(mat.Max(coeffs))
	if coeff == 0 {
		coeff = 1
	}
	for i := range inst.NumSubsets {
		for j := i + 1; j < inst.NumSubsets; j++ {
			intsersectionSize := mat.Dot(inst.Subsets.ColView(i), (inst.Subsets.ColView(j)))
			conflictSize := math.Round(intsersectionSize) - float64(conflictThreshold)
			if conflictSize > eps {
				conflictCost := coeff * conflictSize
				inst.Conflicts.Set(i, j, float64(conflictCost))
				inst.Conflicts.Set(j, i, float64(conflictCost))
				inst.ConflictsList = append(inst.ConflictsList, []int{i, j})
			}
		}
	}
	return nil
}

func LoadInstance(filename string, conflictThreshold int) (*Instance, error) {
	inst := new(Instance)
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	err = errorCoalesce(
		inst.parseFirstLine(scanner),
		inst.parseSecondLine(scanner),
		inst.parseIncompSets(scanner),
		inst.computeConflicts(conflictThreshold),
	)
	if err != nil {
		return nil, err
	}
	return inst, nil
}
