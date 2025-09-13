package main

import (
	"bufio"
	"fmt"
	"log"
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

func (inst *SCPCSInstance) parseFirstLine(scanner *bufio.Scanner) error {
	scanner.Scan()
	line := strings.Fields(scanner.Text())
	numElements, err := strconv.Atoi(line[0])
	if err != nil {
		return fmt.Errorf("Error while parsing first line: %v", err)
	}
	numSubsets, err := strconv.Atoi(line[1])
	if err != nil {
		return fmt.Errorf("Error while parsing first line: %v", err)
	}

	inst.NumElements = numElements
	inst.NumSubsets = numSubsets
	inst.Subsets = mat.NewDense(inst.NumElements, numSubsets, nil)
	inst.Costs = mat.NewVecDense(numSubsets, nil)

	return nil
}

func (inst *SCPCSInstance) parseSecondLine(scanner *bufio.Scanner) error {
	scanner.Scan()
	line := strings.Fields(scanner.Text())
	for i, tok := range line {
		v, err := strconv.Atoi(tok)
		if err != nil {
			return fmt.Errorf("Error while parsing second line: %v", err)
		}
		inst.Costs.SetVec(i, float64(v))
	}
	return nil
}

func (inst *SCPCSInstance) parseIncompSets(scanner *bufio.Scanner) error {
	i := 0
	for scanner.Scan() {
		line := strings.Fields(scanner.Text())[1:]
		for _, tok := range line {
			v, err := strconv.Atoi(tok)
			if err != nil {
				return fmt.Errorf("Error while parsing incompatibility set %d: %v", i, err)
			}
			inst.Subsets.Set(i, v-1, 1)
		}
		i++
	}
	return nil
}

func (inst *SCPCSInstance) computeConflicts(conflictThreshold int) error {
	inst.Conflicts = mat.NewDense(inst.NumSubsets, inst.NumSubsets, nil)
	for i := range inst.NumSubsets {
		for j := i + 1; j < inst.NumSubsets; j++ {
			intsersectionSize := mat.Dot(inst.Subsets.ColView(i), (inst.Subsets.ColView(j)))
			conflictSize := int(math.Max(float64(intsersectionSize)-float64(conflictThreshold), 0))
			if conflictSize > 0 {
				conflictCost := int(math.Round(
					math.Max(
						float64(inst.Costs.At(i, 0))/float64(mat.Sum(inst.Subsets.ColView(i))),
						1.0,
					),
				))
				inst.Conflicts.Set(i, j, float64(conflictCost))
				inst.Conflicts.Set(j, i, float64(conflictCost))
			}
		}
	}
	return nil
}

func LoadInstance(filename string, conflictThreshold int) (*SCPCSInstance, error) {
	inst := new(SCPCSInstance)
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

func main() {
	fmt.Println("Loading instance...")
	instance, err := LoadInstance("data/example.txt", 0)
	if err != nil {
		log.Fatal(err)
	}
	// fmt.Println("Solving instance...")
	// solution, err := instance.Solve()
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// fmt.Println(solution)

	fmt.Println("Lagrangian step...")
	solution, err := instance.SolveWithLagrangianRelaxation()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(solution)
}
