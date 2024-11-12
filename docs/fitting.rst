Fitting Conics
==============

Algebraic Fitting
-----------------

.. autofunction:: conics.fitting.fit_dlt

.. autofunction:: conics.fitting.fit_nievergelt


Geometric Fitting
-----------------

Geometric fitting of conics involves minimizing some sort of orthogonal
distances between the observed points and the points on the conic. This is in
stark contrast to algebraic fitting which generally minimizes the quadratic curve
representation of conics, which can be done linearly. Geometric fitting,
however, generally requires solving a system of non-linear equations.

Solving a system of non-linear equations is typically done iteratively by
minimizing a cost function assumed to be convex. The latter requires an initial
guess of the solution which is reasonably close to the global optimum. In
practice, one can employ algebraic fit to estimate the initial set of the conic
parameters and then refine solution using non-linearly. An example of a such
approach is implemented by :func:`conics.Parabola.refine` and illustrated in
the following example.

.. plot:: ../examples/parabolas.py

   Geometrically fitted parabola in general position.


Parabola to Quadratic Bézier Curve
----------------------------------

A parabola can be exactly represented by a second-degree Bézier curve. Employing
accurate and does not suffer from quantization errors that particularly can
occur when computing isocurves.

Specifying a second-degree Bézier curve requires providing three control points
which determine the curve. While the outer control points can be chosen to be
arbitrarily placed on the curve, the central control point must be computed from
the intersection of the parabola slopes at the outer control points.


.. autofunction:: conics.fitting.parabola_to_bezier

.. plot:: ../examples/bezier.py


The Importance of Normalization
-------------------------------

Standardizing the 2-D coordinates to be mean-free with unit standard deviation
may improve the numerical robustness of the fitting algorithm
:cite:`Chojnacki2003,Harker2004`.

.. plot:: ../examples/normalization.py

   The effect of normalization on (algebraic) fit of a parabola.

In this example, the large range of the vertical axis dominates over the much
smaller range of the horizontal axis. Without scaling, the algebraic fit
therefore produces an elongated parabola that covers the predominant vertical
axis. While this is a valid solution, it is perceptually inferior to the one
estimated using normalized coordinates.


Constraining Conics
-------------------

Sometimes, one wishes to refine an already fitted conic with respect to a set of
points to obtain a conic of specific type or with specific properties. For
instance, the conic obtained using direct linear transform (DLT) results in a
circle. However, one wants a parabola instead. Such constraint can be enforced
using :meth:`conics.Conic.constrain`.
