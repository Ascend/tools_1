import math
import os
import random
import time
import numpy as np

from ais_infer.summary import summary
from ais_infer.utils import (get_file_content, get_file_datasize,
                            get_fileslist_from_dir, list_split, logger,
                            save_data_to_files)

pure_infer_fake_file = "pure_infer_data"
padding_infer_fake_file = "padding_infer_fake_file"

def get_pure_infer_data(size, pure_data_type):
    lst = []
    if pure_data_type == "random":
        # random value from [0, 255]
        lst = [random.randrange(0, 256) for _ in range(size)]
    else:
        # zero value, default
        lst = [0 for _ in range(size)]

    barray = bytearray(lst)
    ndata = np.frombuffer(barray, dtype=np.uint8)
    return ndata

# get numpy array from files list combile all files
def get_narray_from_files_list(files_list, size, pure_data_type, auto_set_dymshape_mode=False):
    ndatalist = []
    for i, file_path in enumerate(files_list):
        logger.debug("get tensor from filepath:{} i:{} of all:{}".format(file_path, i, len(files_list)))
        if file_path == pure_infer_fake_file:
            ndata = get_pure_infer_data(size, pure_data_type)
        elif file_path == padding_infer_fake_file:
            logger.debug("padding file use fileslist[0]:{}".format(files_list[0]))
            ndata = get_file_content(files_list[0])
        elif file_path == None or os.path.exists(file_path) == False:
            logger.error('filepath:{} not valid'.format(file_path))
            raise RuntimeError()
        else:
            ndata = get_file_content(file_path)
        ndatalist.append(ndata)
    if len(ndatalist) == 1:
        return ndatalist[0]
    else:
        ndata = np.concatenate(ndatalist)
        if auto_set_dymshape_mode == False and ndata.nbytes != size:
            logger.error('ndata size:{} not match {}'.format(ndata.nbytes, size))
            raise RuntimeError()
        return ndata

# get tensors from files list combile all files
def get_tensor_from_files_list(files_list, session, size, pure_data_type, auto_set_dymshape_mode=False):
    ndata = get_narray_from_files_list(files_list, size, pure_data_type, auto_set_dymshape_mode)
    tensor = session.create_tensor_from_arrays_to_device(ndata)
    return tensor

# Obtain filesperbatch runcount information according to file information and input description information
# The strategy is as follows:  Judge according to the realsize and file size of input 0. If the judgment fails,
# you need to force the desired value to be set
def get_files_count_per_batch(intensors_desc, fileslist, auto_set_dymshape_mode=False):
    # get filesperbatch
    filesize = get_file_datasize(fileslist[0][0])
    tensorsize = intensors_desc[0].realsize
    if auto_set_dymshape_mode == True:
        files_count_per_batch = 1
    else:
        if filesize == 0 or tensorsize % filesize != 0:
            logger.error('arg0 tensorsize: {} filesize: {} not match'.format(tensorsize, filesize))
            raise RuntimeError()
        files_count_per_batch = (int)(tensorsize/filesize)

    runcount = math.ceil(len(fileslist[0]) / files_count_per_batch)
    #runcount = len(fileslist[0]) // files_count_per_batch
    logger.info("get filesperbatch files0 size:{} tensor0size:{} filesperbatch:{} runcount:{}".format(
        filesize, tensorsize, files_count_per_batch, runcount))
    return files_count_per_batch, runcount

# out api create empty data
def create_intensors_zerodata(session, pure_data_type):
    intensors = []
    intensors_desc = session.get_inputs()
    for info in intensors_desc:
        logger.debug("info shape:{} type:{} val:{} realsize:{} size:{}".format(info.shape, info.datatype, int(info.datatype), info.realsize, info.size))
        ndata = get_pure_infer_data(info.realsize, pure_data_type)
        tensor = session.create_tensor_from_arrays_to_device(ndata)
        intensors.append(tensor)
    return intensors

# Obtain tensor information and files information according to the input filelist. Create intensor form files list
# len(files_list) should equal len(intensors_desc)
def create_infileslist_from_fileslist(fileslist, intensors_desc, auto_set_dymshape_mode=False):
    if len(intensors_desc) != len(fileslist):
        logger.error('fileslist:{} intensor:{} not match'.format(len(fileslist), len(intensors_desc)))
        raise RuntimeError()
    files_count_per_batch, runcount = get_files_count_per_batch(intensors_desc, fileslist, auto_set_dymshape_mode)

    files_perbatch_list = [ list(list_split(fileslist[j], files_count_per_batch, padding_infer_fake_file)) for j in range(len(intensors_desc)) ]

    infileslist = []
    for i in range(runcount):
        infiles = []
        for j in range(len(intensors_desc)):
            logger.debug("create infileslist i:{} j:{} runcount:{} lists:{} filesPerPatch:{}".format(i, j, runcount, files_perbatch_list[j][i], files_count_per_batch))
            infiles.append(files_perbatch_list[j][i])
        infileslist.append(infiles)
    return infileslist

#  outapi. Obtain tensor information and files information according to the input filelist. Create intensor form files list
def create_intensors_from_infileslist(infileslist, intensors_desc, session, pure_data_type, auto_set_dymshape_mode=False):
    intensorslist = []
    for i, infiles in enumerate(infileslist):
        intensors = []
        for j, files in enumerate(infiles):
            tensor = get_tensor_from_files_list(files, session, intensors_desc[j].realsize, pure_data_type, auto_set_dymshape_mode)
            intensors.append(tensor)
        intensorslist.append(intensors)
    return intensorslist

def check_input_parameter(inputs_list, intensors_desc):
    if len(inputs_list) == 0:
        logger.error("Invalid args. Input args are empty")
        raise RuntimeError()
    if os.path.isfile(inputs_list[0]):
        for file_path in inputs_list:
            realpath = os.readlink(file_path) if os.path.islink(file_path) else file_path
            if not os.path.isfile(realpath):
                logger.error("Invalid args. file_path:{} realpath:{} not exist".format(file_path, realpath))
                raise RuntimeError()
    elif os.path.isdir(inputs_list[0]):
        if len(inputs_list) != len(intensors_desc):
            logger.error("Invalid args. args input dir num:{0} not equal to model inputs num:{1}".format(
                len(inputs_list), len(intensors_desc)))
            raise RuntimeError()

        for dir_path in inputs_list:
            real_dir_path = os.readlink(dir_path) if os.path.islink(dir_path) else dir_path
            if not os.path.isdir(real_dir_path):
                logger.error("Invalid args. {} of input args is not a real dir path".format(real_dir_path))
                raise RuntimeError()
    else:
        logger.error("Invalid args. {}  of --input is invalid".format(inputs_list[0]))
        raise RuntimeError()


# outapi. get by input parameters of  inputs_List.
def create_infileslist_from_inputs_list(inputs_list, intensors_desc, auto_set_dymshape_mode=False):
    check_input_parameter(inputs_list, intensors_desc)
    fileslist = []
    inputlistcount = len(inputs_list)
    intensorcount = len(intensors_desc)
    if os.path.isfile(inputs_list[0]) == True:
        chunks = inputlistcount // intensorcount
        fileslist = list(list_split(inputs_list, chunks, padding_infer_fake_file))
        logger.debug("create intensors list file type inlistcount:{} intensorcont:{} chunks:{} files_size:{}".format(
            inputlistcount, intensorcount, chunks, len(fileslist)))
    elif os.path.isdir(inputs_list[0]) and inputlistcount == intensorcount:
        fileslist = [get_fileslist_from_dir(dir) for dir in inputs_list]
        logger.debug("create intensors list dictionary type inlistcount:{} intensorcont:{} files_size:{}".format(
            inputlistcount, intensorcount, len(fileslist)))
    else:
        logger.error('create intensors list filelists:{} intensorcont:{} error create'.format(inputlistcount, intensorcount))
        raise RuntimeError()

    infileslist = create_infileslist_from_fileslist(fileslist, intensors_desc, auto_set_dymshape_mode)
    if len(infileslist) == 0:
        logger.error('create_infileslist_from_fileslist return infileslist size: {}'.format(len(infileslist)))
        raise RuntimeError()

    return infileslist

def save_tensors_to_file(outputs, output_prefix, infiles_paths, outfmt, index, output_batchsize_axis):
    files_count_perbatch = len(infiles_paths[0])
    infiles_perbatch = np.transpose(infiles_paths)
    for i, out in enumerate(outputs):
        ndata = np.array(out)
        if output_batchsize_axis >= len(ndata.shape):
            logger.error("error i:{0} ndata.shape:{1} len:{2} <= output_batchsize_axis:{3}  is invalid".format(
                i, ndata.shape, len(ndata.shape), output_batchsize_axis))
            raise RuntimeError()
        if files_count_perbatch == 1 or ndata.shape[output_batchsize_axis] % files_count_perbatch == 0:
            subdata = np.array_split(ndata, files_count_perbatch, output_batchsize_axis)
            for j in range(files_count_perbatch):
                sample_id = index*files_count_perbatch+j
                #file_path = os.path.join(output_prefix, "input{}_output_{}.{}".format(sample_id, i, outfmt.lower()))
                if infiles_perbatch[j][0] == padding_infer_fake_file:
                    logger.debug("sampleid:{} i:{} infiles:{} is padding fake file so continue".format(
                        sample_id, i, infiles_perbatch[j]))
                    continue
                file_path = os.path.join(output_prefix, "{}_{}.{}".format(
                    os.path.basename(infiles_perbatch[j][0]).split('.')[0], i, outfmt.lower()))
                summary.add_sample_id_infiles(sample_id, infiles_perbatch[j])
                logger.debug("save func: sampleid:{} i:{} infiles:{} outfile:{} fmt:{} axis:{}".format(
                    sample_id, i, infiles_perbatch[j], file_path, outfmt, output_batchsize_axis))
                summary.append_sample_id_outfile(sample_id, file_path)
                save_data_to_files(file_path, subdata[j])
        else:
            logger.error('save out files error array shape:{} filesinfo:{} files_count_perbatch:{} ndata.shape{}:{}'.format(
                ndata.shape, infiles_paths, files_count_perbatch, output_batchsize_axis, ndata.shape[output_batchsize_axis]))
            raise RuntimeError()