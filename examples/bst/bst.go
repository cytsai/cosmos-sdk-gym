package main

import (
	"fmt"
	"time"
	"runtime"
	"math/rand"
)

func printState() {
	fmt.Printf("STATE ")
	pc := make([]uintptr, 32)
	frames := runtime.CallersFrames(pc[:(runtime.Callers(3, pc) - 2)])
	for {
		frame, more := frames.Next()
		fmt.Printf("%s.%d;", frame.Function, frame.Line)
		if !more {
			break
		}
	}
	fmt.Printf("\n")
}

func guidedInt() int {
	printState()
	var rand int
	if _, err := fmt.Scanf("%d\n", &rand); err != nil {
		panic(err)
	}
	fmt.Printf("ACTION %d\n", rand)
	return rand
}

func guidedFloat() float32 {
	printState()
	var rand float32
	if _, err := fmt.Scanf("%g\n", &rand); err != nil {
		panic(err)
	}
	fmt.Printf("ACTION %g\n", rand)
	return rand
}

const (
	BST_DEPTH = 4
	BST_PRUNE = 0.5
	BST_MAXP1 = 32
)

type BST struct {
	value int
	left  *BST
	right *BST
}

func randBST(depth int) *BST {
	bst := &BST{value: guidedInt()} //int(guidedFloat() * BST_MAXP1)}
	if depth < BST_DEPTH {
		if rand.Float32() >= BST_PRUNE {
			bst.left = randBST(depth + 1)
		}
		if rand.Float32() >= BST_PRUNE {
			bst.right = randBST(depth + 1)
		}
	}
	return bst
}

func sprintBST(bst *BST) string {
	if bst == nil {
		return ""
	}
	return fmt.Sprintf("(%s,%d,%s)", sprintBST(bst.left), bst.value, sprintBST(bst.right))
}

func main() {
	rand.Seed(time.Now().UnixNano())
	bst := randBST(0)
	fmt.Printf("DONE %s\n", sprintBST(bst))
}
