
import numpy as np
import cdd
import cvxpy as cp
import pandas as pd
import itertools

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)
def angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def compute_polytope_vertices(A_hat, b_hat):
    """
    This is a copy of https://github.com/stephane-caron/pypoman/blob/master/pypoman/duality.py
    which unfortunately does not like to install with github actions.
   
    Compute the vertices of a polytope given in halfspace representation by
    :math:`A x \\leq b`.
    Parameters
    ----------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.
    Returns
    -------
    vertices : list of arrays
        List of polytope vertices.
 
    """
    b = b_hat.reshape((b_hat.shape[0], 1))
    # np.hstack([-b, A_hat])
    try:
        mat = cdd.matrix_from_array(np.hstack([b, -A_hat]), rep_type=cdd.RepType.INEQUALITY)
        P = cdd.polyhedron_from_matrix(mat)
    except RuntimeError:
        mean, sigma = 0, 0.0001
        A_hat = A_hat + np.random.normal(mean, sigma, size=A_hat.shape) 
        mat = cdd.matrix_from_array(np.hstack([b, -A_hat]), rep_type=cdd.RepType.INEQUALITY)
        P = cdd.polyhedron_from_matrix(mat)
        
    g = cdd.copy_generators(P)
    V = np.array(g.array)
    
    if len(V) == 0:
        mean, sigma = 0, 0.0001
        A_hat = A_hat + np.random.normal(mean, sigma, size=A_hat.shape) 
        mat = cdd.matrix_from_array(np.hstack([b, -A_hat]), rep_type=cdd.RepType.INEQUALITY)
        P = cdd.polyhedron_from_matrix(mat)
        g = cdd.copy_generators(P)
        V = np.array(g.array)
        
    
    vertices = []
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            raise Exception("Polyhedron is not a polytope")
        elif i not in list(g.lin_set):
            vertices.append(V[i, 1:])
    return vertices

def compute_sec(Z_border, net_position, exchange_constraints=None):
    """
    Compute minimal exchanges between zones given net positions and zone borders.

    Parameters
    ----------
    Z_border : pd.DataFrame
        DataFrame (zones × zones) with 1 if a border exists between zones, 0 otherwise.
    net_position : list of float
        Net positions for each zone (same order as Z_border.index).
    exchange_constraints : dict, optional
        Constraints on specific exchanges, e.g. {('Z1', 'Z2'): ('==', 0), ('Z3', 'Z4'): ('<=', 20)}.

    Returns
    -------
    pd.Series
        Series with MultiIndex of (from_zone, to_zone) and exchange values.
    """

    Z = list(Z_border.index)
    n = len(Z)

    EX = cp.Variable((n, n), pos=True)
    SLACK = cp.Variable(n)

    # Objective: minimize total exchange and penalize slack heavily
    obj = cp.Minimize(cp.sum(EX) + 1e5 * cp.sum(cp.abs(SLACK)))

    constraints = []

    # Net position balance per zone
    for i, z in enumerate(Z):
        constraints.append(cp.sum(EX[i, :]) - cp.sum(EX[:, i]) == net_position[i] + SLACK[i])

    # Add automatic zero exchange constraints for non-bordering zones
    for i, from_zone in enumerate(Z):
        for j, to_zone in enumerate(Z):
            if i != j and Z_border.loc[from_zone, to_zone] == 0:
                constraints.append(EX[i, j] == 0)

    # Optional user-defined bilateral exchange constraints
    if exchange_constraints:
        for (from_zone, to_zone), (operator, value) in exchange_constraints.items():
            i = Z.index(from_zone)
            j = Z.index(to_zone)
            if operator == '==':
                constraints.append(EX[i, j] == value)
            elif operator == '<=':
                constraints.append(EX[i, j] <= value)
            elif operator == '>=':
                constraints.append(EX[i, j] >= value)
            else:
                raise ValueError(f"Unsupported operator: {operator}")

    # Solve the problem
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        print("Clarabel failed, trying SCS as fallback...")
        prob.solve(solver=cp.SCS)

    # Create result as Series with MultiIndex
    index = list(itertools.permutations(Z, 2))
    data = np.round([EX.value[Z.index(i[0]), Z.index(i[1])] for i in index], 6)
    exchange = pd.Series(index=index, data=data)

    return exchange