// Copyright 2020 The Orthogonal Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf32"
)

// Random32 return a random float32
func Random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}

func main() {
	rand.Seed(1)

	input, output := tf32.NewV(3, 4), tf32.NewV(6, 4)
	w1, b1 := tf32.NewV(3, 3), tf32.NewV(3)
	w2, b2 := tf32.NewV(6, 6), tf32.NewV(6)
	identity1 := tf32.Identity(3, 3)
	identity2 := tf32.Identity(6, 6)
	parameters := []*tf32.V{&w1, &b1, &w2, &b2}
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, Random32(-1, 1))
		}
	}
	var deltas [][]float32
	for _, p := range parameters {
		deltas = append(deltas, make([]float32, len(p.X)))
	}
	l1 := tf32.Everett(tf32.Add(tf32.Mul(w1.Meta(), input.Meta()), b1.Meta()))
	l2 := tf32.Add(tf32.Mul(w2.Meta(), l1), b2.Meta())
	regularization1 := tf32.Quadratic(identity1.Meta(), tf32.Mul(w1.Meta(), tf32.T(w1.Meta())))
	regularization2 := tf32.Quadratic(identity2.Meta(), tf32.Mul(w2.Meta(), tf32.T(w2.Meta())))
	regularization := tf32.Add(tf32.Sum(regularization1), tf32.Sum(regularization2))
	cost := tf32.Add(tf32.Avg(tf32.Quadratic(l2, output.Meta())), regularization)

	data := [...][3]float32{
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 0},
	}
	for _, item := range data {
		input.X = append(input.X, item[:]...)
		output.X = append(output.X, item[:]...)
		output.X = append(output.X, item[:]...)
	}
	iterations := 100
	alpha, eta := float32(.4), float32(.6)
	points := make(plotter.XYs, 0, iterations)
	for i := 0; i < iterations; i++ {
		for _, p := range parameters {
			p.Zero()
		}
		identity1.Zero()
		identity2.Zero()

		total := tf32.Gradient(cost).X[0]

		norm := float32(0)
		for _, p := range parameters {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 1 {
			scaling := 1 / norm
			for k, p := range parameters {
				for l, d := range p.D {
					deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
					p.X[l] += deltas[k][l]
				}
			}
		} else {
			for k, p := range parameters {
				for l, d := range p.D {
					deltas[k][l] = alpha*deltas[k][l] - eta*d
					p.X[l] += deltas[k][l]
				}
			}
		}

		//fmt.Println(total)
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		if total < .001 {
			fmt.Println("stopping...")
			break
		}
	}

	fmt.Println(w1.X)
	test := tf32.Sum(tf32.Quadratic(tf32.Mul(w1.Meta(), tf32.T(w1.Meta())), identity1.Meta()))
	test(func(a *tf32.V) bool {
		fmt.Println(a.X)
		return true
	})

	fmt.Println(w2.X)
	test = tf32.Sum(tf32.Quadratic(tf32.Mul(w2.Meta(), tf32.T(w2.Meta())), identity2.Meta()))
	test(func(a *tf32.V) bool {
		fmt.Println(a.X)
		return true
	})

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	for i := range data {
		input.X[0], input.X[1], input.X[2] =
			data[i][0], data[i][1], data[i][2]
		var output tf32.V
		l2(func(a *tf32.V) bool {
			output = *a
			return true
		})
		if data[i][2] == 1 && output.X[2] < .5 {
			panic(fmt.Errorf("output should be 1 %f %f %f %f", output.X[0], data[i][0], data[i][1], data[i][2]))
		} else if data[i][2] == 0 && output.X[2] >= .5 {
			panic(fmt.Errorf("output should be 0 %f %f %f %f", output.X[0], data[i][0], data[i][1], data[i][2]))
		}
	}
}
