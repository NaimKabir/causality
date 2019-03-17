import numpy as np
import pandas as pd
from itertools import product
import pulp

def make_partition_probabilities(values):
    
    partition_probabilities = np.array(values)
    partition_probabilities = partition_probabilities / partition_probabilities.sum()

    assert partition_probabilities.sum() == 1
    
    return partition_probabilities

def simulate_participants(partition_probabilities, n_participants, partition_types):

    # drawing participant compliance and response behaviors according to the
    # specified distribution
    participant_partition = np.random.choice(range(len(partition_types)), n_participants, p=partition_probabilities)
    compliance_type, response_type = list(zip(*partition_types[participant_partition]))

    # randomly assigning participants to Control and Treatment groups
    assignments = np.array(['control', 'treatment'])
    participant_assignment = assignments[np.concatenate([np.zeros(n_participants//2), np.ones(n_participants//2)]).astype('int32')]

    # compiling all information into our dataframe
    df = pd.DataFrame({'assignment': participant_assignment,
                       'compliance_type': compliance_type,
                       'response_type': response_type})

    # Simulate outcomes

    # Depending on assignment and compliance type, do you take the treatment?
    df['took_treatment'] = (df.compliance_type == 'always_taker') \
                           | ( (df.compliance_type == 'complier') & (df.assignment == 'treatment')) \
                           | ( (df.compliance_type == 'defier') & (df.assignment == 'control'))

    df.took_treatment = df.took_treatment.astype('int32')

    # Depending on whether you took the treatment and your response_type, what happens to you?
    df['good_outcome'] = (df.response_type == 'always_better') \
                         | ( (df.response_type == 'helped') & (df.took_treatment) )

    df.good_outcome = df.good_outcome.astype('int32')
    
    return df

def get_state_probabilities(df):
    # get all the probabilities we need: p(z,x,y) for each z,x,y combination
    states = list(product(['treatment','control'],[0,1], [0,1]))

    p_states = {f"{assignment}/{'treated' if took == 1 else 'untreated'}/{'good' if outcome ==1 else 'bad'}" : 
                ( (df[df.assignment == assignment].took_treatment == took)
                   & (df[df.assignment == assignment].good_outcome == outcome)  ).mean()
                for assignment, took, outcome in states
                }
    
    return p_states

def set_up_variables(partition_types):
    partition_names = ['/'.join([compliance, response]) for compliance, response in partition_types]
    q = {partition: pulp.LpVariable(partition, lowBound=0) for partition in partition_names}
    return q

def solve(problem, q, p_states):
    
    problem += sum([v for k,v in q.items()]) == 1
    
    p_treatment_untreated_bad = q['never_taker/never_better'] + q['defier/never_better'] \
                             + q['never_taker/helped'] + q['defier/helped']

    p_treatment_untreated_good = q['never_taker/always_better'] + q['defier/always_better'] \
                                 + q['never_taker/hurt'] + q['defier/hurt']

    p_treatment_treated_bad = q['always_taker/never_better'] + q['complier/never_better'] \
                                 + q['always_taker/hurt'] + q['complier/hurt']

    p_treatment_treated_good = q['always_taker/always_better'] + q['complier/always_better'] \
                                 + q['always_taker/helped'] + q['complier/helped']

    p_control_untreated_bad = q['never_taker/never_better'] + q['complier/never_better'] \
                                 + q['never_taker/helped'] + q['complier/helped']

    p_control_untreated_good = q['never_taker/always_better'] + q['complier/never_better'] \
                                 + q['never_taker/hurt'] + q['complier/hurt']

    p_control_treated_bad = q['always_taker/never_better'] + q['defier/never_better'] \
                                 + q['always_taker/hurt'] + q['defier/hurt']

    p_control_treated_good = q['always_taker/always_better'] + q['defier/always_better'] \
                                 + q['always_taker/helped'] + q['defier/helped']
    
    problem += p_treatment_untreated_bad == p_states['treatment/untreated/bad']
    problem += p_treatment_untreated_good == p_states['treatment/untreated/good']
    problem += p_treatment_treated_bad == p_states['treatment/treated/bad']
    problem += p_control_untreated_bad == p_states['control/untreated/bad']
    problem += p_control_untreated_good == p_states['control/untreated/good']
    problem += p_control_treated_bad == p_states['control/treated/bad']
    
    problem += q['complier/helped'] + q['defier/helped'] + q['always_taker/helped'] + q['never_taker/helped'] \
              - q['complier/hurt'] - q['defier/hurt'] - q['always_taker/hurt'] - q['never_taker/hurt']
    
    
    status = pulp.LpStatus[problem.solve()]
    
    if status != 'Optimal':
        raise ValueError('Infeasible')
        
    q_solved = {partition:pulp.value(partition_p) for partition, partition_p in q.items()}
    
    return q_solved

def symbolic_solve(p):
    
    p111 = p['treatment/treated/good']
    p101 = p['treatment/treated/bad']
    p001 = p['treatment/untreated/bad']
    p011 = p['treatment/untreated/good']
    
    p110 = p['control/treated/good']
    p100 = p['control/treated/bad']
    p000 = p['control/untreated/bad']
    p010 = p['control/untreated/good'] 

    
    
    q_min = max(p111 + p000 - 1,
                p110 + p001 - 1,
                p110 - p111 - p101 - p010 - p100,
                p111 - p110 - p100 - p011 - p101,
                -p011 - p101,
                -p010 - p100,
                p001 - p011 - p101 - p010 - p000,
                p000 - p010 - p100 - p011 - p001
               ) 
    
    q_max = min(1 - p011 - p100,
               1 - p010 - p101,
               -p010 + p011 + p001 + p110 + p000,
               -p011 + p111 + p001 + p010 + p000,
               p111 + p001,
               p110 + p000,
               -p101 + p111 + p001 + p110 + p100,
               -p100 + p110 + p000 + p111 + p101)
    
    return q_min, q_max

true_ate = lambda df: (df.response_type == 'helped').mean() - (df.response_type == 'hurt').mean()

itt_ate = lambda df: df[df.assignment=='treatment'].good_outcome.mean()- df[df.assignment=='control'].good_outcome.mean()

q_ate = lambda q: q['complier/helped'] + q['defier/helped'] + q['always_taker/helped'] + q['never_taker/helped'] \
              - q['complier/hurt'] - q['defier/hurt'] - q['always_taker/hurt'] - q['never_taker/hurt']
    