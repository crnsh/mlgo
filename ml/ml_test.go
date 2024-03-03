package ml

import (
	"math"
	"testing"
)

const eps = 0.00001

func floatEquals(a, b, epsilon float64) bool {
	diff := math.Abs(a - b)
	if a == b { // shortcut, handles infinities
		return true
	} else if a == 0 || b == 0 || diff < math.SmallestNonzeroFloat64 {
		// a or b is zero or both are extremely close to it
		// relative error is less meaningful here
		return diff < (epsilon * math.SmallestNonzeroFloat64)
	} else { // use relative error
		return diff/(math.Abs(a)+math.Abs(b)) < epsilon
	}
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

func TestPow(t *testing.T) {

	graph := Graph{ThreadsCount: 1}

	a_data := []float32{4.57, 12, 9.12, 412.324, 6.32}
	b_data := []float32{5.42, 3.44, 5.33, 3, 8.99}

	a := NewTensor1DWithData(nil, TYPE_F32, 5, a_data)
	b := NewTensor1DWithData(nil, TYPE_F32, 5, b_data)
	out := Pow(nil, a, b)

	// run the computation
	BuildForwardExpand(&graph, out)
	GraphCompute(nil, &graph)

	for i := 0; i < len(out.Data); i++ {
		a_val := float64(a.Data[i])
		b_val := float64(b.Data[i])
		out_val := float64(out.Data[i])
		exp_val := math.Pow(a_val, b_val)

		if !floatEquals(out_val, exp_val, eps) {
			t.Errorf("POW(%v, %v) = %v, Expected = %v", a_val, b_val, out_val, exp_val)
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

// VIEW OPS (NOPs)

func TestReshape(t *testing.T) {

	x1 := NewTensor2DWithData(nil, TYPE_F32, 2, 3, []float32{1, 2, 3, 4, 5, 6})
	out := Reshape1D(nil, x1, 6)

	if out.NE[0] != 6 || (out.NE[1]*out.NE[2]*out.NE[3] != 1) {
		t.Errorf("RESHAPE(x1, 6) = %v, Expected = [6,1,1,1]", out.NE)
	}

	x2 := NewTensor2DWithData(nil, TYPE_F32, 6, 2, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	out = Reshape3D(nil, x2, 3, 2, 2)

	if out.NE[0] != 3 || out.NE[1] != 2 || out.NE[2] != 2 || (out.NE[3] != 1) {
		t.Errorf("RESHAPE(x2, 3, 2, 2) = %v, Expected = [3,2,2,1]", out.NE)
	}
}
