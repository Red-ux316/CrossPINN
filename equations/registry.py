from equations.dim1.Volterra.volterra import Problem1, Problem2
from equations.dim1.VolterraFredholm.volterra_fredholm import Problem3, Problem4
from equations.dim1.Abel.abel import Problem5, Problem6
from equations.dim2.Fredholm.fredholm2d import Problem2D_Fredholm1
from equations.dim2.Volterra.volterra2d import Problem2D_Volterra1, Problem8_Volterra2D
from equations.IDE.Volterra.volterra_ide import (
    ProblemIDE_Volterra1, Problem24_VIDE_Nonlinear_1stOrder, 
    Problem25_VIDE_Linear_2ndOrder
)
from equations.Benchmark.benchmark import (
    Problem1_SpectralBias, Problem2_Transient, Problem3_VIDE, Problem19_IDESystem1D, 
    Problem23_HelmholtzFredholm, Problem34_ViscoelasticPDEIDE
)
from equations.Calorimeter.calorimetro import Calorimeter_Exp1, Calorimeter_Exp2, Calorimeter_Exp3, Calorimeter_Exp4

PROBLEMS_MAP = {
    1: Problem1, 
    2: Problem2, 
    3: Problem3, 
    4: Problem4, 
    5: Problem5,
    6: Problem6,
    8: Problem8_Volterra2D, '8_volterra2d': Problem8_Volterra2D,
    11: Problem2D_Fredholm1, 
    12: Problem2D_Volterra1,
    19: Problem19_IDESystem1D, '19': Problem19_IDESystem1D, '19_idesystem': Problem19_IDESystem1D, 'system1d': Problem19_IDESystem1D,
    22: ProblemIDE_Volterra1,
    23: Problem23_HelmholtzFredholm, '23_helmholtz': Problem23_HelmholtzFredholm, 'helmholtz': Problem23_HelmholtzFredholm,
    24: Problem24_VIDE_Nonlinear_1stOrder,
    25: Problem25_VIDE_Linear_2ndOrder,
    31: Problem1_SpectralBias, 32: Problem2_Transient, 33: Problem3_VIDE,
    34: Problem34_ViscoelasticPDEIDE, '34_viscoelastic': Problem34_ViscoelasticPDEIDE, 'viscoelastic': Problem34_ViscoelasticPDEIDE,
    '41_cal_exp1': Calorimeter_Exp1, 41: Calorimeter_Exp1,
    '42_cal_exp2': Calorimeter_Exp2, 42: Calorimeter_Exp2,
    '43_cal_exp3': Calorimeter_Exp3, 43: Calorimeter_Exp3,
    '44_cal_exp4': Calorimeter_Exp4, 44: Calorimeter_Exp4
}

def get_problem(prob_id, a=None, b=None):
    if prob_id not in PROBLEMS_MAP:
        raise ValueError(f"Unknown problem_id: {prob_id}")
    
    if a is not None and b is not None:
        return PROBLEMS_MAP[prob_id](a=a, b=b)
    return PROBLEMS_MAP[prob_id]
