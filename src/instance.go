package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

type Instance struct {
	NumElements       int
	Incompatibilities []*Incompatibility
}

type Incompatibility struct {
	Cost int
	Set  map[int]bool
}

func errorCoalesce(args ...error) error {
	for _, e := range args {
		if e != nil {
			return e
		}
	}
	return nil
}

func (inst *Instance) parseFirstLine(scanner *bufio.Scanner) error {
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
	inst.Incompatibilities = make([]*Incompatibility, numIncomp)

	return nil
}

func (inst *Instance) parseSecondLine(scanner *bufio.Scanner) error {
	scanner.Scan()
	line := strings.Fields(scanner.Text())
	for i, tok := range line {
		v, err := strconv.Atoi(tok)
		if err != nil {
			return fmt.Errorf("Error while parsing second line: %v", err)
		}
		inst.Incompatibilities[i] = &Incompatibility{
			Cost: v,
			Set:  make(map[int]bool),
		}
	}
	return nil
}

func (inst *Instance) parseIncompSets(scanner *bufio.Scanner) error {
	i := 0
	for scanner.Scan() {
		line := strings.Fields(scanner.Text())
		for _, tok := range line {
			v, err := strconv.Atoi(tok)
			if err != nil {
				return fmt.Errorf("Error while parsing incompatibility set %d: %v", i, err)
			}
			inst.Incompatibilities[v-1].Set[i] = true
		}
		i++
	}
	return nil
}

func (inst *Instance) LoadInstance(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	return errorCoalesce(
		inst.parseFirstLine(scanner),
		inst.parseSecondLine(scanner),
		inst.parseIncompSets(scanner),
	)
}

func (inst *Instance) String() string {
	s := new(strings.Builder)
	s.WriteString(fmt.Sprintf("N. elements: %d\n", inst.NumElements))
	s.WriteString(fmt.Sprintf("N. sets: %d\n", len(inst.Incompatibilities)))

	for _, incomp := range inst.Incompatibilities {
		s.WriteString(fmt.Sprintf("Cost: %d, ", incomp.Cost))
		s.WriteString("Elements: ")
		for e, _ := range incomp.Set {
			s.WriteString(fmt.Sprintf("%d ", e))
		}
		s.WriteRune('\n')
	}
	return s.String()
}

func main() {
	instance := new(Instance)
	err := instance.LoadInstance("../data/example2.txt")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(instance)
}
