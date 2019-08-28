import pandas as pd
import numpy as np

start_params = np.asarray([

    ## CCs populations parameters
    [0.1945, 1, 0.05, 0.5],  # 1 r, Viable cancer cells volume doubling time, #0.194
    [1423.1, 0, 1200., 4000.], # 2 k, Tumor carrying capacity
    [0.01678, 0, 0.001, 0.1],  # 3 a, Ccls' killing rate
    [0.26428, 0, 0.05, 1.],   # 4 d, Clearance rate of dying cells

    [0.03, 0, 0.01, 0.1] ,   # 5 l, Decay rate of effector cells
    [0.135329, 1, 0.01, 0.5],   # 6 omega,  Baseline T cell recruitment rate
    [15.3682, 0, 1, 100], # 7 omega2, Fold change in the baseline T cell recruitment rate due to immunogenic cell death
    [8.49542, 0, 1, 50],   # 8 e, Initial fold change in recruitment of cytotoxic T cells caused by immunotherapy
    [0.9667, 0, 0.05, 5],  # 9 clr, 9H10 immunotherapy clearance rate

    #not mentioned parameter
    [3.26, 0, 0.1, 20],  # 10 gęstość efektorów
    [0.01, 0, 0, 1],      # 11 proportion of dead cells in L

    # each parameter is defined as: [nominal value, 1 = fit and 0 = don't fit, lower bound, upper bound]
    ## defining initial guess dor suviving fractions (SFD)
    [0.265, 1, 0, 1],  # 20 Gy
    [0.664, 1, 0, 1],  # 8 Gy
    [0.783, 1, 0, 1],  # 6 Gy

    ## defining initial values for AID
    [0.194, 1, 0, 1],  # 20 Gy
    [0.984, 1, 0, 1],  # 8 Gy
    [0.367, 1, 0, 1]  # 6 Gy

])


def zip_array(left, right, name):
    l, r = np.array(left[name]), np.array(right[name])
    return np.array(list(zip(l, r)))
# [groups, tumor sites, days] = [None, 2, 5]
left = pd.read_csv('data/2/A_Left.csv')
right = pd.read_csv('data/2/A_Right.csv')
right_weight = list(pd.read_csv('data/2/B.csv')['RightY'])

experimental_data_TSA = {
    'TSA_0Gy': {
        'rt': [],  # Radiation therapy, List[Tuple[int, int]], [(day, dose), ...]
        'it': np.array([np.inf]),
        'v': zip_array(left, right, '0Gy+PBSY'),
        'cl': 'TSA',  # Tumor cell line, Str
        'fw': right_weight[0]  # Weight of right tumor at day 35
    },
    'TSA_0Gy+9H10(14,17,20)': {
        'rt': [],                                 # Radiation therapy, List[Tuple[int, int]], [(day, dose), ...]
        'it': np.array([14,17,20]),               # Immunotherapy, Array([int, ...]), [day, ...]
        'v': zip_array(left, right, '0Gy+9H10Y'), # Tumors volume, Array([[int, int], ...]), [[left, right], ...]
        'cl': 'TSA',                              # Tumor cell line, Str
        'fw': right_weight[1]                     # Weight of right tumor at day 35
    },
    'TSA_20Gyx1': {
        'rt': [(12, 20)],
        'it': np.array([np.inf]),
        'v': zip_array(left, right, '20Gyx1+PBSY'),
        'cl': 'TSA',
        'fw': right_weight[2]
    },
    'TSA_20Gyx1+9H10(14,17,20)': {
        'rt': [(12, 20)],
        'it': np.array([14, 17, 20]),
        'v': zip_array(left, right, '20Gyx1+9H10Y'),
        'cl': 'TSA',
        'fw': right_weight[3]

    },
    'TSA_8Gyx3': {
        'rt': [(12, 8), (13, 8), (14, 8)],
        'it': np.array([np.inf]),
        'v' : zip_array(left, right, '8Gyx3+PBSY'),
        'cl': 'TSA',
        'fw': right_weight[4]  # 2x complete tumor regression
    },
#outlayer
    'TSA_8Gyx3+9H10(14,17,20)': {
        'rt': [(12, 8), (13, 8), (14, 8)],
        'it': np.array([14, 17, 20]),
        'v': zip_array(left, right, '8Gyx3+9H10Y'),
        'cl': 'TSA',
        'fw': right_weight[5]  # 2x complete tumor regression
    },
    'TSA_6Gyx5': {
        'it': np.array([np.inf]),
        'rt': [(12, 6), (13, 6), (14, 6), (15, 6), (16, 6)],
        'v' : zip_array(left, right, '6Gyx5+PBSY'),
        'cl': 'TSA',
        'fw': right_weight[6]
    },
    'TSA_6Gyx5+9H10(14,17,20)': {
        'rt': [(12, 6), (13, 6), (14, 6), (15, 6), (16, 6)],
        'it': np.array([14, 17, 20]),
        'v': zip_array(left, right, '6Gyx5+9H10Y'),
        'cl': 'TSA',
        'fw': right_weight[7]
    },
}
left = pd.read_csv('data/3/A_Left.csv')
right = pd.read_csv('data/3/A_Right.csv')

experimental_data_TSA.update(
    {
        'TSA_9H10(12,15,18)': {
            'rt': [],
            'it': np.array([12, 15, 18]),
            'v': zip_array(left, right, '9H10(12-15-18)Y'),
            'cl': 'TSA'
        },
        'TSA_20Gyx1+9H10(12,15,18)': {
            'rt': [(12, 20)],
            'it': np.array([12, 15, 18]),
            'v': zip_array(left, right, 'IR+9H10(12-15-18)Y'),
            'cl': 'TSA'
        }
    }
)
left = pd.read_csv('data/3/B_Left.csv')
right = pd.read_csv('data/3/B_Right.csv')
experimental_data_TSA.update(
    {
        'TSA_8Gyx3+9H10(12,15,18)': {
            'rt': [(12, 8), (13, 8), (14, 8)],
            'it': np.array([12, 15, 18]),
            'v': zip_array(left, right, 'IR+9H10(12-15-18)Y'),
            'cl': 'TSA'
        },
        'TSA_8Gyx3+9H10(16,18,21)': {
            'rt': [(12, 8), (13, 8), (14, 8)],
            'it': np.array([16, 18, 21]),
            'v': zip_array(left, right, 'IR+9H10(16-18-21)Y'),
            'cl': 'TSA'
        }
    }
)
#MCA38
left = pd.read_csv('data/6/A_Left.csv')
right = pd.read_csv('data/6/A_Right.csv')
experimental_data_MCA = {
    'MCA38_0Gy+9H10(14,17,20)': {
            'rt': [],
            'it': np.array([14, 17, 20]),
            'v': zip_array(left, right, '0Gy+9H10Y'),
            'cl': 'MCA38'
        },
        'MCA38_0Gy': {
            'rt': [],
            'it': np.array([np.inf]),
            'v': zip_array(left, right, '0Gy+PBSY'),
            'cl': 'MCA38'
        },
        'MCA38_20Gyx1': {
            'rt': [(12, 20)],
            'it': np.array([np.inf]),
            'v': zip_array(left, right, '20Gyx1+PBSY'),
            'cl': 'MCA38'
        },
        'MCA38_20Gyx1+9H10(14,17,20)': {
            'rt': [(12, 20)],
            'it': np.array([14, 17, 20]),
            'v': zip_array(left, right, '20Gyx1+9H10Y'),
            'cl': 'MCA38'
        },
        'MCA38_8Gyx3': {
            'rt': [(12, 8), (13, 8), (14, 8)],
            'it': np.array([np.inf]),
            'v': zip_array(left, right, '8Gyx3+PBSY'),
            'cl': 'MCA38'
        },
        'MCA38_8Gyx3+9H10(14,17,20)': {
            'rt': [(12, 8), (13, 8), (14, 8)],
            'it': np.array([14, 17, 20]),
            'v': zip_array(left, right, '8Gyx3+9H10Y'),
            'cl': 'MCA38'
        }
}