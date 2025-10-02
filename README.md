# SCPCS with conflicts

Branch and bound algorithm for the Set Covering Problem with conflict sets

## Build

To build the tools use the `go build` utility on `src/scpcs_solve/scpcs_solve.go` and `src/generator/generator.go` to build the solving algorithm and the instance generator, respectively.

## Usage

```
Usage of ./scpcs_solve:
  -highs
        Solve the problem using the HiGHS solver
  -inst value
        a list of instance file paths, separated by a whitespace
  -lagrangean
        Solve with branch and bound using lagrangean relaxation for dual
  -threshold int
        Define the minimum intersection size between subsets to be considered in conflict
```

```
Usage of ./generator:
  -elems int
        The number of elements
  -meand float
        The subsets density mean
  -out string
        The output file (default "out.txt")
  -sets int
        The number of subsets
  -stddevd float
        The subsets density standard deviation
```