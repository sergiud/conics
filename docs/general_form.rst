============
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
