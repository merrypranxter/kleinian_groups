import numpy as np
from typing import Union, Optional


class Mobius:
    """Möbius transformation f(z) = (az+b)/(cz+d), stored as 2×2 complex matrix."""

    def __init__(self, a: complex, b: complex, c: complex, d: complex):
        self._matrix = np.array([[a, b], [c, d]], dtype=np.complex128)

    @property
    def a(self) -> complex:
        return self._matrix[0, 0]

    @property
    def b(self) -> complex:
        return self._matrix[0, 1]

    @property
    def c(self) -> complex:
        return self._matrix[1, 0]

    @property
    def d(self) -> complex:
        return self._matrix[1, 1]

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def __call__(self, z):
        z = np.asarray(z, dtype=np.complex128)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        result = np.empty_like(z)

        if np.abs(self.c) < 1e-300:
            result = (self.a * z + self.b) / self.d
            inf_mask = np.isinf(np.abs(z))
            if np.any(inf_mask):
                result[inf_mask] = complex(np.inf) if np.abs(self.a) > 1e-300 else complex(0)
        else:
            denom = self.c * z + self.d
            pole_mask = np.abs(denom) < 1e-300
            inf_mask = np.isinf(np.abs(z))
            regular_mask = ~pole_mask & ~inf_mask
            result[regular_mask] = (self.a * z[regular_mask] + self.b) / denom[regular_mask]
            result[pole_mask] = complex(np.inf)
            result[inf_mask] = self.a / self.c

        return complex(result[0]) if scalar else result

    def __mul__(self, other: 'Mobius') -> 'Mobius':
        M = self._matrix @ other._matrix
        return Mobius(M[0, 0], M[0, 1], M[1, 0], M[1, 1])

    def inverse(self) -> 'Mobius':
        det = self.a * self.d - self.b * self.c
        return Mobius(self.d / det, -self.b / det, -self.c / det, self.a / det)

    def conjugate_by(self, phi: 'Mobius') -> 'Mobius':
        return phi * self * phi.inverse()

    def fixed_points(self) -> np.ndarray:
        if np.abs(self.c) < 1e-300:
            if np.abs(self.a - self.d) < 1e-300:
                return np.array([complex(np.inf)])
            return np.array([-self.b / (self.a - self.d)])
        A_coef = self.c
        B_coef = self.d - self.a
        C_coef = -self.b
        disc = B_coef ** 2 - 4 * A_coef * C_coef
        sqrt_disc = np.sqrt(complex(disc))
        z1 = (-B_coef + sqrt_disc) / (2 * A_coef)
        z2 = (-B_coef - sqrt_disc) / (2 * A_coef)
        if np.abs(z1 - z2) < 1e-12:
            return np.array([z1])
        return np.array([z1, z2])

    def trace(self) -> complex:
        return self.a + self.d

    def is_parabolic(self) -> bool:
        t2 = self.trace() ** 2
        return np.abs(t2 - 4) < 1e-10

    def is_elliptic(self) -> bool:
        t2 = self.trace() ** 2
        return np.real(t2) < 4 and np.abs(np.imag(t2)) < 1e-10

    def is_hyperbolic(self) -> bool:
        t2 = self.trace() ** 2
        return np.real(t2) > 4 and np.abs(np.imag(t2)) < 1e-10

    def is_loxodromic(self) -> bool:
        t2 = self.trace() ** 2
        return np.abs(np.imag(t2)) > 1e-10

    def isometric_circle(self) -> tuple:
        if np.abs(self.c) < 1e-300:
            raise ValueError("c=0: transformation has no finite isometric circle")
        center = -self.d / self.c
        radius = 1.0 / np.abs(self.c)
        return (center, radius)

    @classmethod
    def from_matrix(cls, M: np.ndarray) -> 'Mobius':
        M = np.asarray(M, dtype=np.complex128)
        return cls(M[0, 0], M[0, 1], M[1, 0], M[1, 1])

    @classmethod
    def identity(cls) -> 'Mobius':
        return cls(1, 0, 0, 1)

    @classmethod
    def parabolic(cls, t: complex) -> 'Mobius':
        return cls(1, t, 0, 1)

    @classmethod
    def hyperbolic(cls, lambda_: complex) -> 'Mobius':
        return cls(lambda_, 0, 0, 1 / lambda_)

    @classmethod
    def rotation(cls, theta: float) -> 'Mobius':
        return cls(np.exp(1j * theta / 2), 0, 0, np.exp(-1j * theta / 2))

    def __repr__(self) -> str:
        return f"Mobius([[{self.a:.4f}, {self.b:.4f}], [{self.c:.4f}, {self.d:.4f}]])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mobius):
            return NotImplemented
        M1 = self._matrix
        M2 = other._matrix
        diff1 = np.max(np.abs(M1 - M2))
        diff2 = np.max(np.abs(M1 + M2))
        return min(diff1, diff2) < 1e-10
