import visualize
import datasets
import ldle
import pickle
import os

use_LLR = True

if use_LLR:
    save_dir_root = 'D:/pyLDLE_LLR/'
else:
    save_dir_root = 'D:/pyLDLE/'


def run_example(example, no_gamma = False):
    print(example)
    save_path = save_dir_root+'/'+example+'.dat'
    if os.path.exists(save_path):
        print(save_path, 'already exists.')
        return
    if example == 'square':
        X, labelsMat, ddX = datasets.Datasets().rectanglegrid(ar=1)
        ldle_obj = ldle.LDLE(X=X, eta_min=5, max_iter0=25,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/square'),
                             vis_y_options = {'cmap0':'summer',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'rectangle':
        X, labelsMat, ddX = datasets.Datasets().rectanglegrid()
        ldle_obj = ldle.LDLE(X=X, eta_min=5, 
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/rectangle'),
                             vis_y_options = {'cmap0':'summer',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'barbell':
        X, labelsMat, ddX = datasets.Datasets().barbell()
        ldle_obj = ldle.LDLE(X=X, eta_min=10,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/barbell'),
                             vis_y_options = {'cmap0':'summer',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'squarewithtwoholes':
        X, labelsMat, ddX = datasets.Datasets().squarewithtwoholes()
        ldle_obj = ldle.LDLE(X=X, eta_min=8,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/squarewithtwoholes'),
                             vis_y_options = {'cmap0':'summer',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'spherewithhole':
        X, labelsMat, ddX = datasets.Datasets().spherewithhole()
        ldle_obj = ldle.LDLE(X=X, eta_min=8,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             max_iter0 = 40,
                             vis = visualize.Visualize(save_dir_root+'/spherewithhole'),
                             vis_y_options = {'cmap0':'hsv',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'swissrollwithhole':
        X, labelsMat, ddX = datasets.Datasets().swissrollwithhole()
        ldle_obj = ldle.LDLE(X=X, eta_min=10, max_iter0=80,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/swissrollwithhole'),
                             vis_y_options = {'cmap0':'summer',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'noisyswissroll':
        X, labelsMat, ddX = datasets.Datasets().noisyswissroll()
        ldle_obj = ldle.LDLE(X=X, eta_min=25,  N=25,
                             no_gamma = no_gamma, max_iter0=40, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/noisyswissroll'),
                             vis_y_options = {'cmap0':'jet',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'sphere':
        X, labelsMat, ddX = datasets.Datasets().sphere()
        ldle_obj = ldle.LDLE(X=X, eta_min=8, max_iter0=80,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/sphere'),
                             vis_y_options = {'cmap0':'hsv',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'flattorus4d':
        X, labelsMat, ddX = datasets.Datasets().flattorus4d()
        ldle_obj = ldle.LDLE(X=X, eta_min=10,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/flattorus4d'),
                             vis_y_options = {'cmap0':'hsv',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'curvedtorus3d':
        X, labelsMat, ddX = datasets.Datasets().curvedtorus3d()
        ldle_obj = ldle.LDLE(X=X, eta_min=15, max_iter0=80,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/curvedtorus3d'),
                             vis_y_options = {'cmap0':'summer',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'kleinbottle4d':
        X, labelsMat, ddX = datasets.Datasets().kleinbottle4d()
        ldle_obj = ldle.LDLE(X=X, eta_min=20, max_iter0=40,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/kleinbottle4d'),
                             vis_y_options = {'cmap0':'hsv',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'mobiusstrip3d':
        X, labelsMat, ddX = datasets.Datasets().mobiusstrip3d()
        ldle_obj = ldle.LDLE(X=X, eta_min=10,
                             no_gamma = no_gamma, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/mobiusstrip3d'),
                             vis_y_options = {'cmap0':'hsv',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    elif example == 'floor':
        X, labelsMat, ddX = datasets.Datasets().floor()
        ldle_obj = ldle.LDLE(X=X, eta_min=5, k=17, max_iter0=40,
                             no_gamma = no_gamma, to_tear = False, use_LLR = use_LLR,
                             vis = visualize.Visualize(save_dir_root+'/floor'),
                             vis_y_options = {'cmap0':'hsv',
                                              'cmap1':'jet',
                                              'labels':labelsMat[:,0]})
    ldle_obj.fit()
    ldle_obj.y_final_ltsa = ldle_obj.compute_final_global_embedding_ltsap_based()
    with open(save_path, "wb") as f:
        pickle.dump([X, labelsMat, ldle_obj], f)

examples = ['square', 'rectangle', 'barbell', 'squarewithtwoholes', 'spherewithhole', 'swissrollwithhole',
            'noisyswissroll', 'sphere', 'flattorus4d', 'curvedtorus3d', 'kleinbottle4d', 'mobiusstrip3d', 'floor']

for example in examples:
    run_example(example)