# interpolation.py
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse


def read_points():
    n = int(input("Enter number of points (n): "))
    x = []
    y = []
    print("Enter x values separated by spaces:")
    x = list(map(float, input().split()))
    print("Enter y values separated by spaces:")
    y = list(map(float, input().split()))
    if len(x) != n or len(y) != n:
        raise ValueError("Number of x or y values does not match n.")
    return x, y, n


def create_matrix(n, x, y):
    matrix = []
    b = []
    for i in range(n):
        row = []
        for j in range(n):
            x_value = x[i] ** j
            row.append(x_value)
        matrix.append(row)
        b.append(y[i])
    return matrix, b


def gaussian_elimination(matrix, b):
    n = len(b)
    for i in range(n):
        # Pivoting
        max_row = max(range(i, n), key=lambda r: abs(matrix[r][i]))
        if i != max_row:
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            b[i], b[max_row] = b[max_row], b[i]

        # Elimination
        for j in range(i + 1, n):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, n):
                matrix[j][k] -= factor * matrix[i][k]
            b[j] -= factor * b[i]

    # Back substitution
    a = [0] * n
    for i in range(n - 1, -1, -1):
        a[i] = (b[i] - sum(matrix[i][j] * a[j] for j in range(i + 1, n))) / matrix[i][i]
    return a


def print_matrix(matrix, b):
    for row in matrix:
        print(" ".join(f"{value:f}" for value in row))

    print("b:", " ".join(f"{value:f}" for value in b))


class Graph:
    @staticmethod
    def evaluate_polynomial(coefficients, x_value):
        """
        Avalia o polinômio nos pontos x_value usando os coeficientes
        P(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
        """
        result = 0
        for i, coef in enumerate(coefficients):
            result += coef * (x_value**i)
        return result

    @staticmethod
    def plot_interpolated_function(x_points, y_points, coefficients):
        """
        Plota o gráfico da função polinomial interpolada
        """
        # Criar um range de valores x para plotar a curva suave
        x_min = min(x_points) - 1
        x_max = max(x_points) + 1
        x_plot = np.linspace(x_min, x_max, 1000)

        # Calcular os valores y correspondentes usando a função polinomial
        y_plot = [Graph.evaluate_polynomial(coefficients, x) for x in x_plot]

        # Criar o gráfico
        plt.figure(figsize=(10, 6))

        # Plotar a função interpolada
        plt.plot(x_plot, y_plot, "b-", linewidth=2, label="Função Interpolada")

        # Plotar os pontos originais
        plt.scatter(
            x_points, y_points, color="red", s=100, zorder=5, label="Pontos Originais"
        )

        # Configurar o gráfico
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Interpolação Polinomial")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Adicionar equação do polinômio no gráfico
        equation = "P(x) = "
        for i, coef in enumerate(coefficients):
            if i == 0:
                equation += f"{coef:.6f}"
            else:
                sign = "+" if coef >= 0 else "-"
                coef_abs = abs(coef)
                if i == 1:
                    equation += f" {sign} {coef_abs:.6f}x"
                else:
                    equation += f" {sign} {coef_abs:.6f}x^{i}"

        plt.text(
            0.02,
            0.98,
            equation,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig("interpolation_plot.png", dpi=300, bbox_inches="tight")
        plt.show()
        print(f"\nGráfico salvo como 'interpolation_plot.png'")

    @staticmethod
    def lagrange_polynomial(x_points, y_points, x_value):
        """
        Avalia o polinômio de Lagrange nos pontos x_value dados os pontos (x_points, y_points)
        """
        n = len(x_points)
        result = 0
        for i in range(n):
            term = y_points[i]
            for j in range(i):
                term *= (x_value - x_points[j]) / (x_points[i] - x_points[j])
            for j in range(i + 1, n):
                term *= (x_value - x_points[j]) / (x_points[i] - x_points[j])
            result += term
        return result

    @staticmethod
    def plot_lagrange(x_points, y_points):
        """
        Plota o gráfico do polinômio de Lagrange
        """
        x_min = min(x_points) - 1
        x_max = max(x_points) + 1
        x_plot = np.linspace(x_min, x_max, 1000)
        y_plot = [Graph.lagrange_polynomial(x_points, y_points, x) for x in x_plot]

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, "g-", linewidth=2, label="Lagrange Interpolação")
        plt.scatter(
            x_points, y_points, color="red", s=100, zorder=5, label="Pontos Originais"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Interpolação Polinomial de Lagrange")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("lagrange_plot.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolation")
    parser.add_argument(
        "--polinomial", action="store_true", help="Use polynomial interpolation"
    )
    parser.add_argument(
        "--lagrange", action="store_true", help="Use Lagrange interpolation"
    )

    args = parser.parse_args()

    # Check if no args provided
    if not (args.polinomial or args.lagrange):
        print("No arguments provided. Please provide --polinomial or --lagrange.")
        sys.exit(1)

    x, y, n = read_points()
    if args.polinomial:
        matrix, b = create_matrix(n, x, y)
        print("\nMatrix: ")
        print_matrix(matrix, b)
        coefficients = gaussian_elimination(matrix, b)
        print("\nCoefficients: ")
        print(" ".join(f"{coef:f}" for coef in coefficients))
    elif args.lagrange:
        coefficients = []
        coefficients.extend(y)

    interpolated_function = []
    interpolated_function.extend(coefficients)

    # Plotar o gráfico da função interpolada
    if args.polinomial:
        Graph.plot_interpolated_function(x, y, coefficients)
    elif args.lagrange:
        Graph.plot_lagrange(x, y)
