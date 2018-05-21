import socket
import os
import h5py
import tensorflow as tf

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

def parse_run_variables(metric,set_mem=False,set_mem_value = 0.50):

    return_parse_dict = {}

    n_gpus = 1
    guill = False
    use_basis_dict = True

    if (socket.gethostname() == 'puck'):
        print "############################################"
        print "DETECTED RUN ON PUCK"
        print "############################################"
        if(set_mem):
            import tensorflow as tf
            from keras.backend.tensorflow_backend import set_session

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = set_mem_value
            set_session(tf.Session(config=config))

        data_store_suffix = '/usr/local/data/sejacob/ANOMALY/densecub'
        use_basis_dict = False

    elif ('gpu' in socket.gethostname()):
        print "############################################"
        print "DETECTED RUN ON HELIOS: Probably"
        print "############################################"
        verbose = 1
        os.chdir('/scratch/suu-621-aa/ANOMALY/densecub')
        data_store_suffix = '/scratch/suu-621-aa/ANOMALY/densecub'

    else:
        print socket.gethostname()
        print "############################################"
        print "DETECTED RUN ON GUILLIMIN: Probably"
        print "############################################"
        verbose = 1
        os.chdir('/gs/project/suu-621-aa/sejacob/densecub')
        data_store_suffix = '/gs/scratch/sejacob/densecub'
        guill = True

        if ('-ngpu' in metric.keys()):
            n_gpus = int(metric['-ngpu'])

    if (guill and '-ngpu' in metric.keys()):
        batch_size = 256 * n_gpus
    else:
        batch_size = 256

    return_parse_dict['n_gpus'] = n_gpus
    return_parse_dict['data_store_suffix']=data_store_suffix
    return_parse_dict['guill'] = guill
    return_parse_dict['batch_size'] = batch_size

    n_chapters = 0
    gs = False
    nic = 0
    tstrides = 4
    lassign = 0.0
    h_units = int(metric['-h'])
    loss = 'dssim'
    ntrain = int(metric['-ntrain'])
    nclusters = int(metric['-nclust'])
    nocl = bool(int(metric['-nocl']))


    return_parse_dict['gs']=gs
    return_parse_dict['lassign'] = lassign
    return_parse_dict['tstrides'] = tstrides
    return_parse_dict['h_units'] = h_units
    return_parse_dict['ntrain'] = ntrain
    return_parse_dict['nclusters'] = nclusters
    return_parse_dict['nocl']=nocl
    return_parse_dict['loss']=loss


    if (nocl):
        suffix = 'nocl_tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(
            n_chapters) + '_clusters_' + str(nclusters)
        lamda = 0.0
    else:
        suffix = 'tstrd_' + str(tstrides) + '_nic_' + str(nic) + '_chapters_' + str(n_chapters) + '_clusters_' + str(
            nclusters)
        lamda = float(metric['-lamda'])

    return_parse_dict['lamda'] = lamda


    suffix += '_hunits_' + str(h_units)

    if (gs):

        train_folder = os.path.join(data_store_suffix, 'chapter_store_conv',
                                    'triangle_data_store_greyscale_bkgsub' + str(tstrides))
        test_data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test',
                                       'triangle_data_store_greyscale_test_bkgsub' + str(tstrides))
        nc = 1

    else:

        train_folder = os.path.join(data_store_suffix, 'chapter_store_conv',
                                    'triangle_data_store_bksgub' + str(tstrides))
        test_data_store = os.path.join(data_store_suffix, 'chapter_store_conv_test',
                                       'triangle_data_store_test_bkgsub' + str(tstrides))
        nc = 3

    return_parse_dict['nc']= nc
    return_parse_dict['train_folder']=train_folder
    return_parse_dict['test_data_store'] = test_data_store

    # Open train dset
    train_dset = h5py.File(os.path.join(train_folder, 'data_train.h5'), 'r')

    if (n_chapters == 0):
        n_chapters = len(train_dset.keys())

    if (nic == 0):
        nic = n_chapters

    if (gs):
        suffix += '_greyscale_'
    else:
        suffix += '_color_'

    return_parse_dict['nic'] = nic
    return_parse_dict['n_chapters']=n_chapters

    print "############################"
    print "SET UP MODEL"
    print "############################"

    if ('-large' in metric.keys()):
        large = bool(int(metric['-large']))
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print "MODEL SIZE LARGE? :", large
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    else:
        large = True

    if ('-tlm' in metric.keys()):
        tlm = metric['-tlm']
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        print "USE TEST LOSS METRIC:", tlm
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    else:
        tlm = 'dssim'

    suffix += '_large_' + str(large)
    suffix += '_ntrain_' + str(ntrain)
    suffix += '_lamda_' + str(lamda)

    # Get MODEL
    model_store = 'models/' + suffix



    return_parse_dict['model_store']=model_store
    return_parse_dict['tlm']=tlm
    return_parse_dict['large']=large
    return_parse_dict['nic']=nic
    return_parse_dict['n_chapters']=n_chapters
    return_parse_dict['nc'] = nc

    return_parse_dict['reverse'] = False
    return_parse_dict['use_basis_dict'] = use_basis_dict

    train_dset.close()

    print return_parse_dict
    return return_parse_dict
