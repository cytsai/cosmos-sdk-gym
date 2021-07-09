package main

import (
	"os"
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

var TREE_DEPTH int
var TREE_PRUNE float32

type Tree struct {
	value int
	left  *Tree
	right *Tree
}

func randTree(depth int) *Tree {
	tree := &Tree{value: guidedInt()}
	if depth < TREE_DEPTH {
		if rand.Float32() >= TREE_PRUNE {
			tree.left = randTree(depth + 1)
		}
		if rand.Float32() >= TREE_PRUNE {
			tree.right = randTree(depth + 1)
		}
	}
	return tree
}

func sprintTree(tree *Tree) string {
	if tree == nil {
		return ""
	}
	return fmt.Sprintf("(%s,%d,%s)", sprintTree(tree.left), tree.value, sprintTree(tree.right))
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Sscanf(os.Args[1], "%d", &TREE_DEPTH)
	fmt.Sscanf(os.Args[2], "%f", &TREE_PRUNE)
	fmt.Printf("DONE %s\n", sprintTree(randTree(0)))
}
