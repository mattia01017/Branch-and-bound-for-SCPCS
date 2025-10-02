package main

import (
	"flag"
	"fmt"
	"os"
	"scp_with_conflicts/src/scpcs_solve/scpcs"
	"strings"
)

func main() {
	var solveHighs, solveLagrangean bool
	var conflictThreshold int
	var paths []string

	flag.Func("inst", "a list of instance file paths, separated by a whitespace", func(s string) error {
		paths = strings.Fields(s)
		return nil
	})
	flag.BoolVar(&solveHighs, "highs", false, "Solve the problem using the HiGHS solver")
	flag.BoolVar(&solveLagrangean, "lagrangean", false, "Solve with branch and bound using lagrangean relaxation for dual")
	flag.IntVar(&conflictThreshold, "threshold", 0, "Define the minimum intersection size between subsets to be considered in conflict")

	flag.Parse()

	if len(paths) == 0 {
		fmt.Fprintln(os.Stderr, "Must specify at least a path")
		os.Exit(1)
	}
	if !solveHighs && !solveLagrangean {
		fmt.Fprintln(os.Stderr, "Must specify a solving algorithm")
		os.Exit(1)
	}

	for _, p := range paths {
		inst, err := scpcs.LoadInstance(p, conflictThreshold)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error for instance \"%v\": %v. Skipping...\n", p, err)
			continue
		}

		if solveHighs {
			fmt.Printf("Solving %v...\n", p)
			sol, err := inst.Solve()
			if err != nil {
				fmt.Fprintf(os.Stderr, "An error occured while solving with HiGHS instance \"%v\": %v\n", p, err)
			} else {
				fmt.Printf("Instance %v:\n%v\n", p, sol)
			}
		}
		if solveLagrangean {
			fmt.Printf("Solving %v...\n", p)
			sol, err := inst.SolveWithLagrangeanRelaxation()
			if err != nil {
				fmt.Fprintf(os.Stderr, "An error occured while solving with B&B instance \"%v\": %v\n", p, err)
			} else {
				fmt.Printf("Instance %v:\n%v\n", p, sol)
			}
		}
		fmt.Println()
	}
}
