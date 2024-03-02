package ml

import (
	"math"
	"testing"
)

const eps = 0.001

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
		exp_val := math.Sqrt(x_val)

		if !floatEquals(exp_val, out_val, eps) {
			t.Errorf("SQRT(%v) = %v, Expected = %v", x_val, out_val, exp_val)
		}
	}
}

func TestDiv(t *testing.T) {

	graph := Graph{ThreadsCount: 1}

	a_data := []float32{1523, 12524, 1823, 142.324, 91238.233}
	b_data := []float32{10.42, 12.44, 65.33, 5, 8.99}

	a := NewTensor1DWithData(nil, TYPE_F32, 5, a_data)
	b := NewTensor1DWithData(nil, TYPE_F32, 5, b_data)
	out := Div(nil, a, b)

	// run the computation
	BuildForwardExpand(&graph, out)
	GraphCompute(nil, &graph)

	for i := 0; i < len(out.Data); i++ {
		a_val := float64(a.Data[i])
		b_val := float64(b.Data[i])
		out_val := float64(out.Data[i])
		exp_val := a_val / b_val

		if !floatEquals(out_val, exp_val, eps) {
			t.Errorf("DIV(%v, %v) = %v, Expected = %v", a_val, b_val, out_val, exp_val)
		}
	}
}

func TestErf(t *testing.T) {

	graph := Graph{ThreadsCount: 1}

	slice := []float32{4.23, 8.32, 3.53, 40.1223, 5512.9152}

	x := NewTensor1DWithData(nil, TYPE_F32, 5, slice)
	out := Erf(nil, x)

	// run the computation
	BuildForwardExpand(&graph, out)
	GraphCompute(nil, &graph)

	for i := 0; i < len(out.Data); i++ {
		x_val := float64(x.Data[i])
		out_val := float64(out.Data[i])
		exp_val := math.Erf(x_val)

		if !floatEquals(exp_val, out_val, eps) {
			t.Errorf("ERF(%v) = %v, Expected = %v", x_val, out_val, exp_val)
		}
	}
}
