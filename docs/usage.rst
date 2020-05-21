=====
Usage
=====

To use Conics in a project::

    import conics


General Form
============

There are several ways of representing a conic section. A common representation,
however, is to employ a 6-D vector of coefficients
:math:`\vec\theta=(A,B,C,D,E,F)^\top\in\mathbb{R}^6` that define the
inhomogeneous quadratic equation

.. math::

    Q(x,y)=Ax^2+Bxy+Cx^2+Dx+Ey+C=0

The quadratic form of the equation can alternatively be defined in terms of a
dot product between the coefficients :math:`\vec\theta` and the monomials in
:math:`x,y` given by the vector :math:`\vec\xi = (x^2,xy,y^2,x,y,1)^\top` as
:cite:`Kanatani2016`

.. math::

    \langle \vec\xi,\vec\theta \rangle = 0
    \enspace .

:math:`\vec\xi` is termed dual-Grassmanian and :math:`\vec\theta` Grassmanian
coordinates of the conics :cite:`Harker2008`.

The quadratic equation can be homogenized be substituting :math:`x'=\frac{x}{w}`
and :math:`y'=\frac{y}{w}` for :math:`(x,y)^\top` :cite:`Hartley2004`. This
allows to obtain a symmetric :math:`3\times3` matrix that describes the conic.
Projective transformations can then be conveniently expressed by matrix
products.

.. autoclass:: conics.Conic
    :members:


Parabola
--------

A parabola is a specific instance of a conic section.

.. autoclass:: conics.Parabola
   :members:


Fitting
=======

The Importance of Normalization
-------------------------------

Standardizing the 2-D coordinates to be mean-free with unit standard deviation
may improve the numerical robustness of the fitting algorithm
:cite:`Chojnacki2003,Harker2004`.

.. plot:: ../examples/normalization.py

   The effect of normalizati on (algebraic) fit of a parabola.

In this example, the large range of the vertical axis dominates over the much
smaller range of the horizontal axis. Without scaling, the algebraic fit
therefore produces an elongated parabola that covers the predominant vertical
axis. While this is a valid solution, it is perceptually inferior to the one
estimated using normalized coordinates.


Algebraic Fitting
-----------------

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


Pose Estimation
===============

.. autofunction:: conics.estimate_pose

.. bibliography:: references.bib
   :style: plain
