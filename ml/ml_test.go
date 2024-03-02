package ml

import (
	"math"
	"testing"
)

const eps = 0.00001

func floatEquals(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestSqrt(t *testing.T) {

	graph := Graph{ThreadsCount: 1}

	slice := []float32{4.23, 8.32, 3.53, 40.1223, 5512.9152}

	x := NewTensor1DWithData(nil, TYPE_F32, 5, slice)
	out := Sqrt(nil, x)

	// run the computation
	BuildForwardExpand(&graph, out)
	GraphCompute(nil, &graph)

	for i := 0; i < len(out.Data); i++ {
		x_val := float64(x.Data[i])
		out_val := float64(out.Data[i])
		x_val_sqrt := math.Sqrt(float64(x.Data[i]))

		if !floatEquals(math.Sqrt(x_val), out_val, eps) {
			t.Errorf("Sqrt(%v) = %v, want %v", x_val, out_val, x_val_sqrt)
		}
	}
}
