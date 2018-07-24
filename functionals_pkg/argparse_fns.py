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
    godiva = False
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

    elif ('godiva' in socket.gethostname() or 'soma' in socket.gethostname()):
        print "############################################"
        print "DETECTED RUN ON GODIVA"
        print "############################################"
        verbose = 1
        data_store_suffix = '/usr/local/data/sejacob/densecub'
        use_basis_dict = False
        godiva = True

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

    if('-dataset' in metric.keys()):
        dataset = metric['-dataset']

    return_parse_dict['n_gpus'] = n_gpus
    return_parse_dict['data_store_suffix']=data_store_suffix
    return_parse_dict['guill'] = guill
    return_parse_dict['batch_size'] = batch_size

    size = 48
    min_data_threshold = 20000
    ntrain = 200
    patience = 15

    if(dataset=='UCSD2'):
        path_to_videos_test='/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
        if(guill):
            path_to_videos_test = '/gs/project/suu-621-aa/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
        if(godiva):
            path_to_videos_test = '/usr/local/data/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

        tstrides = 1
        sp_strides = 12
        greyscale=True

    if(dataset=='UCSD1'):
        path_to_videos_test='/usr/local/data/sejacob/ANOMALY/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
        if(guill):
            path_to_videos_test = '/gs/project/suu-621-aa/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test'
        if(godiva):
            path_to_videos_test = '/usr/local/data/sejacob/data/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

        tstrides = 1
        sp_strides = 12
        greyscale = True

    elif(dataset=='TRIANGLE'):
        path_to_videos_test = '/usr/local/data/sejacob/ANOMALY/data/art_videos_triangle/Test'
        tstrides = 4
        sp_strides = 7
        greyscale = False
        size = 24

    elif(dataset=='BOAT-HOLBORN'):
        path_to_videos_test = '/usr/local/data/sejacob/ANOMALY/data/york/Boat-Holborn/Test'
        tstrides = 2
        sp_strides = 12
        greyscale = False
        min_data_threshold = 3000
        ntrain = 300
        patience = 15

    elif(dataset=='BOAT-SEA'):
        path_to_videos_test = '/usr/local/data/sejacob/ANOMALY/data/york/Boat-Sea/Test'
        tstrides = 4
        sp_strides = 12
        greyscale = False
        min_data_threshold = 3000
        ntrain = 300
        patience = 15

    elif(dataset=='CAMOUFLAGE'):
        path_to_videos_test = '/usr/local/data/sejacob/ANOMALY/data/york/Camouflage/Test'
        tstrides = 1
        sp_strides = 12
        greyscale = False
        min_data_threshold = 3000
        ntrain = 300
        patience = 15

    elif(dataset=='CANOE'):
        path_to_videos_test = '/usr/local/data/sejacob/ANOMALY/data/york/Canoe/Test'
        tstrides = 2
        sp_strides = 12
        greyscale = False
        min_data_threshold = 3000
        ntrain = 300
        patience = 15

    elif(dataset=='TRAFFIC-TRAIN'):
        path_to_videos_test = '/usr/local/data/sejacob/ANOMALY/data/york/Traffic-Train/Test'
        tstrides = 4
        sp_strides = 12
        greyscale = False
        min_data_threshold = 3000
        ntrain = 300
        patience = 15

    n_chapters = 0
    gs = greyscale
    nic = 0
    lassign = 0.0
    h_units = int(metric['-h'])
    loss = 'dssim'
    nclusters = int(metric['-nclust'])
    nocl = bool(int(metric['-nocl']))


    return_parse_dict['gs']=gs
    return_parse_dict['lassign'] = lassign
    return_parse_dict['tstrides'] = tstrides
    return_parse_dict['sp_strides'] = sp_strides
    return_parse_dict['h_units'] = h_units
    return_parse_dict['ntrain'] = ntrain
    return_parse_dict['nclusters'] = nclusters
    return_parse_dict['nocl']=nocl
    return_parse_dict['loss']=loss
    return_parse_dict['size']=size
    return_parse_dict['min_data_threshold']=min_data_threshold
    return_parse_dict['patience']=patience

    if (nocl):
        suffix = 'nocl_tstrd_' + str(tstrides) + '_clusters_' + str(nclusters)
        lamda = 0.0
    else:
        suffix = 'tstrd_' + str(tstrides) + '_clusters_' + str(nclusters)
        lamda = float(metric['-lamda'])

    return_parse_dict['lamda'] = lamda


    suffix += '_hunits_' + str(h_units)


    train_folder = os.path.join(data_store_suffix, 'DATA',dataset,'TRAIN')
    test_data_store = os.path.join(data_store_suffix, 'DATA',dataset,'TEST')

    if (gs):
        nc = 1

    else:
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
    suffix += '_lamda_' + str(lamda)

    # Get MODEL
    model_store = dataset + '_models/' + suffix



    return_parse_dict['model_store']=model_store
    return_parse_dict['tlm']=tlm
    return_parse_dict['large']=large
    return_parse_dict['nic']=nic
    return_parse_dict['n_chapters']=n_chapters
    return_parse_dict['nc'] = nc

    return_parse_dict['reverse'] = False
    return_parse_dict['use_basis_dict'] = use_basis_dict
    return_parse_dict['path_to_videos_test'] = path_to_videos_test

    train_dset.close()

    print return_parse_dict
    return return_parse_dict
