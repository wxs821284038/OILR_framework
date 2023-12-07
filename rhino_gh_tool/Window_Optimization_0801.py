import h5py
import glob
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skopt import gp_minimize
from keras_dgl.layers import MultiGraphCNN
from Window_Optimization_Module_0512 import *

# These functions define the optimization problem, it relocate specified windows while keeping their dimension
def define_varibles(rm_param, open_param, win_ids, win_ind_op, res):
    sorter = np.argsort(win_ids)
    row_inds = sorter[np.searchsorted(win_ids, win_ind_op, sorter=sorter)].tolist()
    curr_pos = []
    hv_bnd = []
    for ind in row_inds:
        xl, xh, yl, yh, zl, zh, rm1, rm2 = open_param[ind]
        r1_xl, r1_xh, r1_yl, r1_yh, r1_zl, r1_zh = 0.0, float("inf"), 0.0, float("inf"), 0.0, 3.0
        r2_xl, r2_xh, r2_yl, r2_yh, r2_zl, r2_zh = 0.0, float("inf"), 0.0, float("inf"), 0.0, 3.0
        if rm1 >= 0:
            r1_xl, r1_xh, r1_yl, r1_yh, r1_zl, r1_zh = rm_param[int(rm1)]            
        if rm2 >= 0:
            r2_xl, r2_xh, r2_yl, r2_yh, r2_zl, r2_zh = rm_param[int(rm2)]
        x_lbnd, x_hbnd = max(r1_xl, r2_xl), min(r1_xh, r2_xh)
        y_lbnd, y_hbnd = max(r1_yl, r2_yl), min(r1_yh, r2_yh)
        z_lbnd, z_hbnd = max(r1_zl, r2_zl), min(r1_zh, r2_zh)
        if xl == xh:
            width = abs(yh-yl)
            h_low, h_high = y_lbnd, y_hbnd - width
            h_curr = yl
        else:
            width = abs(xh-xl)
            h_low, h_high = x_lbnd, x_hbnd - width
            h_curr = xl
        height = abs(zh-zl)
        v_low, v_high = z_lbnd, z_hbnd - height
        v_curr = zl
        curr_pos += [h_curr/res, v_curr/res]
        hv_bnd += [(h_low/res, h_high/res), (v_low/res, v_high/res)]
    return np.array(curr_pos).astype(int), np.array(hv_bnd).astype(int)

def update_rm_param(rm_param, open_param, win_ids, win_ind_op, res, x):
    """
    x: a list / 1D vector of windows' int position on a vertical wall in format of [w1_hpos, w1_vpos, w2_hpos, w2_vpos ......]
    """
    row_inds = np.searchsorted(win_ids, win_ind_op).tolist()
    new_open_param = open_param.copy()
    
    for i, ind in enumerate(row_inds):
        hpos, vpos = x[i*2]*res, x[i*2+1]*res
        xl, xh, yl, yh, zl, zh, rm1, rm2 = open_param[ind]
        height = abs(zh-zl)
        zl = vpos
        zh = vpos + height
        if xl == xh:
            width = abs(yh-yl)
            yl, yh = hpos, hpos + width
        else:
            width = abs(xh-xl)
            xl, xh = hpos, hpos + width
        new_open_param[ind] = [xl, xh, yl, yh, zl, zh, rm1, rm2]
    
    new_params = (rm_param, new_open_param)
    return new_params

def airflow_pattern_evaluation(flow_pattern, rm_param, low_thres, up_thres, ext_nxy, ext_nz, res):
    """
    Term A is the summation of undesired velocity potions in 5 rooms. Undesired 
    velocity is defined as less than 0.1m/s (low_thres) or above 0.5m/s (up_thres) 
    and the position should be below 2m in vertical dimension.
    """
    x0_min = min([float(rm_param[i][0]) for i in range(len(rm_param))])
    x1_max = max([float(rm_param[i][1]) for i in range(len(rm_param))])
    y0_min = min([float(rm_param[i][2]) for i in range(len(rm_param))])
    y1_max = max([float(rm_param[i][3]) for i in range(len(rm_param))])
    x_length, y_length= abs(x1_max-x0_min), abs(y1_max-y0_min)
    
    x_s = x0_min - round((ext_nxy-x_length/res)/3.0)*res + res/2.0
    y_s = y0_min - round((ext_nxy-y_length/res)/2.0)*res + res/2.0
    z_s = 0.0 + res/2.0
    
    proportions = []
    for i in range(len(rm_param)):
        rxl, rxh, ryl, ryh, rzl, rzh = rm_param[i]
        ixl, ixh, iyl, iyh, izl, izh = int((rxl+res/2.0-x_s)/res), int((rxh+res/2.0-x_s)/res),             int((ryl+res/2.0-y_s)/res), int((ryh+res/2.0-y_s)/res), int((rzl+res/2.0-z_s)/res), int((rzh+res/2.0-z_s)/res)
        rm_i = flow_pattern[ixl:ixh, iyl:iyh, izl:izl+8, :] # position should be below 2m in vertical dimension
        rm_mag = np.sqrt(np.square(rm_i[:,:,:,0])+np.square(rm_i[:,:,:,1])+np.square(rm_i[:,:,:,2]))
        prop = np.sum((rm_mag<low_thres)|(rm_mag>up_thres))/rm_mag.size
        # prop = np.sum((rm_mag<low_thres))/rm_mag.size
        proportions.append(prop.round(6))
    
    term_a = np.sum(proportions).round(6)
    print("rm prop: " + str(proportions) + " | sum Prop: " + str(term_a))
    return term_a

def objective_func(x):
    delta = 0.25
    ext_nxy, ext_nz, nxy, nz = 128, 32, 32, 16
    low_thres = 0.1
    up_thres = 0.5
    new_rm_op_param = update_rm_param(rm_param, open_param, win_ids, win_ind_op, delta, x)
    predicted_flow_pattern = predict_one_case(new_rm_op_param, GCN_model, Outdoor_model, In_mag_model, In_hres_model, Regularizer_model, ext_nxy, ext_nz, nxy, nz, delta)
    loss = airflow_pattern_evaluation(predicted_flow_pattern, rm_param, low_thres, up_thres, ext_nxy, ext_nz, delta)
    return loss

if __name__ == '__main__':
    delta = 0.25
    
    root_path = "C:/Users/Xiaoshi/Dropbox/9_PhD_Semester_9/01_PhD_Research/10_Grasshopper_Tooling/wind_driven/"
    rm_param, open_param, win_ids = get_room_parameters(root_path)
    # these indices of windows are based on Grasshopper indexing of window openings.
    win_ind_op = [3, 5, 6, 8]

    # load GCN model
    tf.autograph.set_verbosity(0)
    GCN_filename = root_path + "models/GCN_model_0316_1300.h5"
    GCN_model = load_model(GCN_filename, custom_objects={'MultiGraphCNN': MultiGraphCNN}, compile=False)

    # load Outdoor model
    Outdoor_filename = root_path + "models/Outdoor_model_0801_000085.h5"
    Outdoor_model = load_model(Outdoor_filename, compile=False)

    # load Indoor model
    In_mag_filename = root_path + "models/Indoor_model_mag_0801_000280.h5"
    In_mag_model = load_model(In_mag_filename, compile=False)

    In_hres_filename = root_path + "models/Indoor_model_uvwp_0801_000225.h5"
    In_hres_model = load_model(In_hres_filename, compile=False)

    # load Regularize model
    Regularizer_filename = root_path + "models/regularizer_gen_model_0801_000058.h5"
    Regularizer_model = load_model(Regularizer_filename, compile=False)

    curr_pos, hv_bnd = define_varibles(rm_param, open_param, win_ids, win_ind_op, delta)
    start = time.time()
    res = gp_minimize(objective_func,            # the function to minimize
                      hv_bnd,                    # [(-2.0, 2.0)], the bounds on each dimension of x
                      acq_func = "EI",           # the acquisition function
                      n_calls = 20,              # the number of evaluations of f
                      n_initial_points = 5,      # the number of random initialization points
                      n_restarts_optimizer = 5,  
                      acq_optimizer = "lbfgs",
                      noise = "gaussian", 
                      random_state = 42,
                      n_jobs = 1,
                      x0 = list(curr_pos))
    print("Case finished optimization in " + str(int(time.time()-start)) +" seconds =========================")
    op_param_optimal = update_rm_param(rm_param, open_param, win_ids, win_ind_op, delta, res.x)[1]
    optimal_op_file = root_path + "opening_optimal.csv"
    np.savetxt(optimal_op_file, op_param_optimal, fmt = "%10.2f", delimiter=",")
    print("Optimizated Window Opening Saved in Local Drive =========================")