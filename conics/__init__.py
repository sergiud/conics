
# conics - Python library for dealing with conics
#
# Copyright 2024 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-level package for Conics."""

__author__ = """Sergiu Deitsch"""
__email__ = 'sergiu.deitsch@gmail.com'
__version__ = '0.1.0'

from ._conic import concentric_conics_vanishing_line  # noqa: F401
from ._conic import Conic  # noqa: F401
from ._conic import estimate_pose  # noqa: F401
from ._conic import projected_center  # noqa: F401
from ._conic import surface_normal  # noqa: F401
from ._ellipse import Ellipse  # noqa: F401
from ._parabola import Parabola  # noqa: F401
