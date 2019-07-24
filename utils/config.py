#-*- encoding:utf8 -*-                                                                                                                   


class DefaultConfig(object):

    acid_one_hot = [0 for i in range(20)]
    acid_idex = {j:i for i,j in enumerate("ACDEFGHIKLMNPQRSTVWY")}


    BASE_PATH = "../../"
    sequence_path = "{0}/data_cache/sequence_data".format(BASE_PATH)
    pssm_path = "{0}/data_cache/pssm_data".format(BASE_PATH)
    dssp_path = "{0}/data_cache/dssp_data".format(BASE_PATH)

    max_sequence_length = 500
    windows_size = 3               

    batch_size = 32
    seq_dim = 20
    dssp_dim = 9
    pssm_dim = 20
   
    kernels = [13,15,17]
    dropout =0.2
    splite_rate = 0.9


