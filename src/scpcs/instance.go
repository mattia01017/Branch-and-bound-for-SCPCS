package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	mapset "github.com/deckarep/golang-set/v2"
)

type SCPCSInstance struct {
	NumElements int
	Subsets     []*Subset
	Conflicts   [][]int
}

type Subset struct {
	Cost int
	Set  mapset.Set[int32]
}

type SCPCSSolution struct {
	SelectedSubsets []int
	TotalCost       int
}

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
	numIncomp, err := strconv.Atoi(line[1])
	if err != nil {
		return fmt.Errorf("Error while parsing first line: %v", err)
	}

	inst.NumElements = numElements
	inst.Subsets = make([]*Subset, numIncomp)

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
		inst.Subsets[i] = &Subset{
			Cost: v,
			Set:  mapset.NewSet[int32](),
		}
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
			inst.Subsets[v-1].Set.Add(int32(i + 1))
		}
		i++
	}
	return nil
}

func (inst *SCPCSInstance) computeConflicts(conflictThreshold int) error {
	inst.Conflicts = make([][]int, len(inst.Subsets))
	for i := range inst.Conflicts {
		inst.Conflicts[i] = make([]int, len(inst.Subsets))
	}
	for i := range inst.Subsets {
		for j := i + 1; j < len(inst.Subsets); j++ {
			intsersectionSize := inst.Subsets[i].Set.Intersect(inst.Subsets[j].Set).Cardinality()
			conflictSize := int(math.Max(float64(intsersectionSize)-float64(conflictThreshold), 0))
			if conflictSize > 0 {
				conflictCost := int(math.Round(
					math.Max(
						float64(inst.Subsets[i].Cost)/float64(inst.Subsets[i].Set.Cardinality()),
						1.0,
					),
				))
				inst.Conflicts[i][j] = conflictCost
				inst.Conflicts[j][i] = conflictCost
			}
		}
	}
	return nil
}

func (sol *SCPCSSolution) String() string {
	s := new(strings.Builder)
	s.WriteString(fmt.Sprintf("Total cost: %d\n", sol.TotalCost))
	s.WriteString(fmt.Sprintf("Selected subsets: %v\n", sol.SelectedSubsets))
	return s.String()
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

func (inst *SCPCSInstance) String() string {
	s := new(strings.Builder)
	s.WriteString(fmt.Sprintf("N. elements: %d\n", inst.NumElements))
	s.WriteString(fmt.Sprintf("N. sets: %d\n", len(inst.Subsets)))

	for _, incomp := range inst.Subsets {
		s.WriteString(fmt.Sprintf("Cost: %d, ", incomp.Cost))
		s.WriteString("Elements: ")
		for e := range incomp.Set.Iter() {
			s.WriteString(fmt.Sprintf("%d ", e))
		}
		s.WriteRune('\n')
	}
	// s.WriteString("Conflicts:\n")
	// for _, row := range inst.Conflicts {
	// 	s.WriteString(fmt.Sprintf("%v\n", row))
	// }
	return s.String()
}

func main() {
	instance, err := LoadInstance("data/example.txt", 0)
	if err != nil {
		log.Fatal(err)
	}
	// fmt.Println(instance)
	solution, err := instance.Solve()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(solution)
}
