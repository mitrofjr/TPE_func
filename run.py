import sys
import time
import random
import pickle
import itertools
import logging
import argparse
import uuid
from math import exp
from multiprocessing import Pool
from utils import format_xyz, tmer2_gmtkn_parser, get_charges_multiplicities
from plots import learning_curve, shap_analysis
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from hyperopt.pyll import scope
from pyscf import dft
from pyscf import gto


def calc_E(params, geom, basis, charge, multiplicity, tmp_id):

    mol = gto.M(atom=geom, basis=basis, charge=charge, spin=multiplicity)
    mr = dft.UKS(mol)
    mr.init_guess = tmp_id
    mr.max_cycle = 1000
    mr.xc = '{}*SR_HF({}) + {}*LR_HF({}) + {}*XC_LDA_X + ' \
            '{}*XC_GGA_X_B88 + {}*XC_GGA_X_PBE + {}*XC_GGA_X_B86 + ' \
            '{}*XC_GGA_X_PW91 + {}*XC_GGA_X_SOGGA, {}*XC_LDA_C_XALPHA + ' \
            '{}*XC_LDA_C_VWN + {}*XC_LDA_C_RPA + {}*XC_GGA_C_PBE + ' \
            '{}*XC_GGA_C_PW91 + {}*XC_GGA_C_LYP'.format(
        params['param_00'], params['param_01'], params['param_02'],
        params['param_01'], params['param_03'], params['param_04'],
        params['param_05'], params['param_06'], params['param_07'],
        params['param_08'], params['param_09'], params['param_10'],
        params['param_11'], params['param_12'], params['param_13'],
        params['param_14'], params['param_15'],
    )

    E = mr.kernel()

    return E


def calc_mol(geom, params, charge, multiplicity):
    tmp_id = 'tmp/' + str(uuid.uuid4())
    
    mol = gto.M(atom=geom, basis='ccpvdz', charge=charge, spin=multiplicity)
    mf = dft.UKS(mol)
    mf.xc = 'PBE'
    mf.chkfile = tmp_id
    mf.kernel()
   
    basis2 = 'augccpvdz'
    basis3 = 'augccpvtz'

    E2 = calc_E(params, geom, basis2, charge, multiplicity, tmp_id)
    E3 = calc_E(params, geom, basis3, charge, multiplicity, tmp_id)
    
    
    if E3 < E2:
        E_total = E3 - 0.42*(E2 - E3)
        logging.info('D-zeta E = {0:7.2f}, T-zeta E = {1:7.2f}, CBS E = {2:7.2f}'.format(E2, E3, E_total))
    
    else:
        logging.warning("No CBS here")
        E_total = E2

    return E_total


def calc_reaction(list_of_geoms, list_of_coefficients, E_real, charges, multiplicities, reaction, params):

    E = 0
    pool = Pool(4)
    list_of_energies = list(pool.starmap(
        calc_mol, zip(list_of_geoms, itertools.repeat(params), charges, multiplicities)))
    pool.close()
    pool.join()
        
    for i in range(len(list_of_energies)):
        E += list_of_coefficients[i] * list_of_energies[i]

    dE = abs(E - E_real)

    logging.info("Reaction: {}".format(reaction))
    logging.info("Calculated value: {:.4f}".format(E))
    logging.info("Reference value: {:.4f}".format(E_real))
         
    return dE, reaction


def hyperopt_search(dataset, param_grid, n_iter, sample, trials_init):
    
    if trials_init:
        with open(trials_init, 'rb') as f:
            trials = pickle.load(f)
            n_trials_init = len(trials.trials)
    else:
        trials = Trials()
        n_trials_init = 0

    def objective(params):
        all_data = tmer2_gmtkn_parser("datasets/{}".format(dataset))
        sampled_data = random.sample(all_data, sample)
        sampled_systems = [k["atoms"] for k in sampled_data]
        sampled_stoichiometry = [k["stoichiometry"] for k in sampled_data]
        sampled_reference_value = [k["reference_value"] / 627.509 for k in sampled_data]
        sampled_charges = [k["charges"] for k in sampled_data]
        sampled_multiplicities = [k["multiplicities"] for k in sampled_data]
        sampled_reactions = [k["reaction"] for k in sampled_data]

        logging.info("New iteration...")
        start = time.time()
        
        losses = []
        reactions = []
            
        for syst, stoich, ref_v, charges, multi, reacs in zip(sampled_systems, sampled_stoichiometry,
                                                              sampled_reference_value, sampled_charges,
                                                              sampled_multiplicities, sampled_reactions):
            output = calc_reaction(syst, stoich, ref_v, charges, multi, reacs, params)
            losses.append(output[0])
            reactions.append(output[1])

        reactions_losses = dict(zip(reactions, losses))

        logging.info("Total loss: {:.4f}".format(sum(losses)))
        logging.info("Iteration took {:.1f} s\n".format(time.time() - start))
        
        return {'loss': sum(losses) / sample, 'params': params, 'reactions_losses': reactions_losses, 'status': STATUS_OK}
    
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    trials_filename = "{0}_{1}_iters_{2}_in_batch_{3}.pickle".format(dataset, n_iter, sample, current_time)

    for n in range(n_trials_init + 1, n_iter + 1):
        try:
            fmin(objective, param_grid, algo=tpe.suggest, trials=trials, max_evals=n)
            with open(trials_filename, 'wb') as f:
                pickle.dump(trials, f)
        except IndexError as err:
            print(err)
            with open(trials_filename, 'rb') as f:
                trials = pickle.load(f)
    
    return trials_filename
                
         


HYPERPARAMETERS = {
    'param_00': hp.uniform('param_00', 0.0, 1.0),
    'param_01': hp.uniform('param_01', 0.0, 1.0),
    'param_02': hp.uniform('param_02', 0.0, 1.0),
    'param_03': hp.uniform('param_03', -1.0, 1.0),
    'param_04': hp.uniform('param_04', -1.0, 1.0),
    'param_05': hp.uniform('param_05', -1.0, 1.0),
    'param_06': hp.uniform('param_06', -1.0, 1.0),
    'param_07': hp.uniform('param_07', -1.0, 1.0),
    'param_08': hp.uniform('param_08', -1.0, 1.0),
    'param_09': hp.uniform('param_09', -1.0, 1.0),
    'param_10': hp.uniform('param_10', -1.0, 1.0),
    'param_11': hp.uniform('param_11', 0.0, 1.0),
    'param_12': hp.uniform('param_12', -1.0, 1.0),
    'param_13': hp.uniform('param_13', -1.0, 1.0),
    'param_14': hp.uniform('param_14', -1.0, 1.0),
    'param_15': hp.uniform('param_15', -1.0, 1.0),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('-i', '--iters', type=int, default=100)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-t', '--trials', default=None)
    args = parser.parse_args()

    logfile = "{0}_{1}_iters_{2}_in_batch_{3}.log".format(
        args.dataset, args.iters, args.batch, time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime()))

    logging.basicConfig(
        filename=logfile, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    trials_filename = hyperopt_search(args.dataset, HYPERPARAMETERS, args.iters, args.batch, args.trials)
    learning_curve(trials_filename, title=args.dataset)
    shap_analysis(trials_filename, title=args.dataset)
