#Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import tempfile
import random
import numpy as np
import scipy.io
from threading import Thread
from flask.ext.socketio import SocketIO, emit, join_room, leave_room, \
    close_room, disconnect

import flask
from flask import Response
import werkzeug.exceptions
import numpy as np
from google.protobuf import text_format
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

import digits
from digits.config import config_value
from digits import utils
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import app, scheduler, autodoc
from digits.model import tasks
from forms import PretrainedFeatureExtractionModelForm
from job import FeatureExtractionModelJob
from digits.status import Status

NAMESPACE   = '/models/images/extraction'

@app.route(NAMESPACE + '/new', methods=['GET'])
@autodoc('models')
def feature_extraction_model_new():
    """
    Return a form for a new FeatureExtractionModelJob for feature extraction.
    """
    form = PretrainedFeatureExtractionModelForm()
    return flask.render_template('models/images/extraction/new.html',
            form = form,
            )

############################################################
# Code for downloading model from gist
############################################################
import os
import yaml
import urllib
import hashlib
import time
import sys

required_keys = ['caffemodel', 'caffemodel_url', 'sha1']

def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def parse_readme_frontmatter(dirname):
    readme_filename = os.path.join(dirname, 'readme.md')
    with open(readme_filename) as f:
        lines = [line.strip() for line in f.readlines()]
    top = lines.index('---')
    bottom = lines[top + 1:].index('---')
    frontmatter = yaml.load('\n'.join(lines[top + 1:bottom]))
    assert all(key in frontmatter for key in required_keys)
    return dirname, frontmatter

def valid_dirname(dirname):
    try:
        return parse_readme_frontmatter(dirname)
    except Exception as e:
        print('ERROR: {}'.format(e))
        raise argparse.ArgumentTypeError(
            'Must be valid Caffe model directory with a correct readme.md')

def download_model(gist_location):
    dirname, frontmatter = valid_dirname(gist_location)
    model_filename = os.path.join(dirname, frontmatter['caffemodel'])

    if os.path.exists(model_filename) and model_checks_out():
         print("Model already exists.")
         return 1

    def model_checks_out(filename=model_filename, sha1=frontmatter['sha1']):
        with open(filename, 'r') as f:
            return hashlib.sha1(f.read()).hexdigest() == sha1

    urllib.urlretrieve(frontmatter['caffemodel_url'], model_filename, reporthook)
    if not model_checks_out():
        print('ERROR: model did not download correctly! Run this again.')
        return -1
    else:
        return 0

def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        time.sleep(10)
        count += 1
        socketio.emit('my response',
                      {'data': 'Server generated event', 'count': count},
                      namespace=NAMESPACE)

# GLOBAL
thread = None

######################################################################################

@app.route(NAMESPACE + '.json', methods=['POST'])
@app.route(NAMESPACE, methods=['POST'])
@autodoc(['models', 'api'])
def feature_extraction_model_create():
    """
    Create a new FeatureExtractionModelJob for feature extraction using pretrained model.

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = PretrainedFeatureExtractionModelForm()

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('models/images/extraction/new.html',
                    form = form,
                    ), 400

    job = None
    try:
        job = FeatureExtractionModelJob(
                name = form.model_name.data,
                )
        network = caffe_pb2.NetParameter()
        pretrained_model = None

        digits_cwd = os.getcwd()
       
        if form.gist_id.data:
            gist_id = form.gist_id.data
            import subprocess

            # Stores the gist in DIGITS_HOME/pretrained_models/gist_id/gist_id_master folder.
            command = digits_cwd+'/scripts/download_model_from_gist.sh '+gist_id
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            process.wait()
            for line in process.stdout:
                print line,
            print "Gist successfully downloaded"

            gist_location = digits_cwd+'/pretrained_models/'+gist_id+'/'+gist_id+'-master'
            ###########
            # MOHIT 
            fl = download_model(gist_location)
            if fl != 0:
                if fl == 1:
                    print "Model already exists"
                else:
                    print "Model download failed"
                raise werkzeug.exceptions.BadRequest('Failed to download caffemodel binary file')
            #################
            # Now download the .caffemodel file from the gist readme.
            #command= digits_cwd+'/scripts/download_model_binary.py '+gist_location
            #process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            #process.wait()
            #for line in process.stdout:
            #    print line,
            #if not process.returncode == 1:
            #    print "Caffe model successfully downloaded from the caffe zoo."
            #else:
            #    raise werkzeug.exceptions.BadRequest('Failed to download caffemodel binary file')
            #################
            
            for filename in os.listdir(gist_location):
                if filename.endswith('.caffemodel'):
                    pretrained_model = str(gist_location+'/'+filename).strip()

            if not pretrained_model:
                raise werkzeug.exceptions.BadRequest('Failed to download caffemodel from gist! : %s' % gist_id)
        
        elif form.caffezoo_model.data:
            import subprocess
            
            model_gist_location = digits_cwd+'/pretrained_models/'+form.caffezoo_model.data
            ###########
            # MOHIT
            global thread
            if thread is None:
                thread = Thread(target=background_thread)
                thread.start()

            fl = download_model(model_gist_location)
            if fl != 0:
                if fl == 1:
                    print "Model already exists"
                else:
                    print "Model download failed"
                raise werkzeug.exceptions.BadRequest('Failed to download caffemodel binary file')

            ################# 
            #command = digits_cwd+'/scripts/download_model_binary.py '+model_gist_location
            #process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            #process.wait()
            #for line in process.stdout:
            #    print line,
            #if not process.returncode == 1:
            #    print "Caffe model successfully loaded from the caffe zoo."
            #else:
            #    raise werkzeug.exceptions.BadRequest('Failed to download caffemodel binary file from zoo.')
            #################

            for filename in os.listdir(model_gist_location):
                if filename.endswith('.caffemodel'):
                    pretrained_model = str(model_gist_location+'/'+filename).strip()

            try:
                with open(model_gist_location+'/deploy.prototxt', 'r') as deploy_file:
                    deploy_content = deploy_file.read()
            except:
               raise werkzeug.exceptions.BadRequest('deploy.prototxt file does not exist in : %s' % model_gist_location) 

            if not pretrained_model or not deploy_content:
                raise werkzeug.exceptions.BadRequest('Model not Found : %s' % form.caffezoo_model.data)

        else:
            try:
                pretrained_model = form.custom_network_snapshot.data.strip()
            except:
                raise werkzeug.exceptions.BadRequest('File does not exist : %s' % form.custom_network_snapshot.data.strip())
        
        if not form.caffezoo_model.data:
            try:
                with open(form.custom_network.data, 'r') as deploy_file:
                    deploy_content = deploy_file.read()
            except:
                raise werkzeug.exceptions.BadRequest('deploy.prototxt file does not exist : %s' % form.custom_network.data)

        if form.method.data == 'custom':
            text_format.Merge(deploy_content, network)
        else:
            raise werkzeug.exceptions.BadRequest(
                    'Unrecognized method: "%s"' % form.method.data)
        
        if form.mean_file.data:
            mean_file = form.mean_file.data
        else:
            mean_file = None

        job.tasks.append(
                tasks.CaffeLoadModelTask(
                    job_dir         = job.dir(),
                    pretrained_model= pretrained_model,
                    crop_size       = None,
                    channels        = None,
                    network         = network,
                    mean_file       = mean_file,
                    )
                )

        scheduler.add_job(job)
        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('models_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

def show(job):
    """
    Called from digits.model.views.models_show()
    """
    import caffe
    
    job_id = job.id()
    model_file = './digits/jobs/'+job_id+'/snapshot_iter_1.caffemodel'
    prototxt_file = './digits/jobs/'+job_id+'/deploy.prototxt'

    meta_data = {}
    try:
        net = caffe.Net(prototxt_file, model_file, caffe.TEST)
        meta_data['InputDimensions'] = net.blobs['data'].data.shape
        meta_data['#Categories'] = net.blobs['prob'].data.shape[1]
    except:
        # wait for the model to be loaded onto memory.
        import time
        time.sleep(2)
        net = caffe.Net(prototxt_file, model_file, caffe.TEST)
        meta_data['InputDimensions'] = net.blobs['data'].data.shape
        meta_data['#Categories'] = net.blobs['prob'].data.shape[1]

    return flask.render_template('models/images/extraction/show.html', job=job, meta_data=meta_data)

@app.route(NAMESPACE + '/large_graph', methods=['GET'])
@autodoc('models')
def feature_extraction_model_large_graph():
    """
    Show the loss/accuracy graph, but bigger
    """
    job = job_from_request()

    return flask.render_template('models/images/extraction/large_graph.html', job=job)

@app.route(NAMESPACE + '/classify_one.json', methods=['POST'])
@app.route(NAMESPACE + '/classify_one', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def feature_extraction_model_classify_one():
    """
    Classify one image and return the top 5 classifications

    Returns JSON when requested: {predictions: {category: confidence,...}}
    """
    job = job_from_request()

    image = None
    if 'image_url' in flask.request.form and flask.request.form['image_url']:
        image = utils.image.load_image(flask.request.form['image_url'])
    elif 'image_file' in flask.request.files and flask.request.files['image_file']:
        with tempfile.NamedTemporaryFile() as outfile:
            flask.request.files['image_file'].save(outfile.name)
            image = utils.image.load_image(outfile.name)
    else:
        raise werkzeug.exceptions.BadRequest('Must provide image_url or image_file')

    # resize image
    model_task = job.load_model_task()
    if model_task.crop_size:
        height = model_task.crop_size
        width = model_task.crop_size
    else:
        raise werkzeug.exceptions.BadRequest('Failed to extract crop_size from network')
    image = utils.image.resize_image(image, height, width,
            #TODO : Get channels elegantly.
            channels = 3,
            resize_mode = None,
            )

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        if 'select_visualization_layer' in flask.request.form and flask.request.form['select_visualization_layer']:
            layers = flask.request.form['select_visualization_layer']
        else:
            layers = 'all'

    #vis_json = False
    #if 'visualization_json' in flask.request.form and flask.request.form['visualization_json']:
    #    vis_json = True

    save_vis_file = False
    save_file_type = ''
    save_vis_file_location = ''
    if 'save_vis_file' in flask.request.form and flask.request.form['save_vis_file']:
        save_vis_file = True
        if 'save_type_mat' in flask.request.form and flask.request.form['save_type_mat']:
            save_file_type = 'mat'
        elif 'save_type_numpy' in flask.request.form and flask.request.form['save_type_numpy']:
            save_file_type = 'numpy'
        else:
            raise werkzeug.exceptions.BadRequest('No filetype selected. Expected .npy or .mat')
        if 'save_vis_file_location' in flask.request.form and flask.request.form['save_vis_file_location']:
            save_vis_file_location = flask.request.form['save_vis_file_location']

    if 'job_id' in flask.request.form and flask.request.form['job_id']:
        job_id = flask.request.form['job_id']
    elif 'job_id' in flask.request.args and flask.request.args['job_id']:
        job_id = flask.request.args['job_id']
    else:
        raise werkzeug.exceptions.BadRequest('job_id is a necessary parameter, not found.')
    
    predictions, visualizations = model_task.infer_one(image, snapshot_epoch=epoch, layers=layers)
    # take top 5
    predictions = [(p[0], round(100.0*p[1],2)) for p in predictions[:5]]

    if save_vis_file:
        if save_file_type == 'numpy':
            try:
                np.array(visualizations).dump(open(save_vis_file_location+'/visualization_'+job_id+'.npy', 'wb'))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as Numpy array')
        elif save_file_type == 'mat':
            try:
                scipy.io.savemat(save_vis_file_location+'/visualization_'+job_id+'.mat', {'visualizations':visualizations})
            except IOError as e:
                raise werkzeug.exceptions.BadRequest('I/O error{%s}: %s'% (e.errno, e.strerror))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as .mat file')
        else:
            raise werkzeug.exceptions.BadRequest('Invalid filetype for visualization data saving')

    if request_wants_json():
        if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
            # flask.jsonify has problems creating JSON from numpy.float32
            # convert all non-dict, non-list and non-string elements to string.
            for layer in visualizations:
                for ele in layer:
                    if not isinstance(layer[ele], dict) and not isinstance(layer[ele], str) and not isinstance(layer[ele], list):
                        layer[ele] = str(layer[ele]) 
            return flask.jsonify({'predictions': predictions, 'visualizations': visualizations})
        else:
            return flask.jsonify({'predictions': predictions})
    else:
        return flask.render_template('models/images/extraction/classify_one.html',
                image_src       = utils.image.embed_image_html(image),
                predictions     = predictions,
                visualizations  = visualizations,
                )

@app.route(NAMESPACE + '/classify_many.json', methods=['POST'])
@app.route(NAMESPACE + '/classify_many', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def feature_extraction_model_classify_many():
    """
    Classify many images and return the top 5 classifications for each

    Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}
    """
    job = job_from_request()

    image_list = flask.request.files['image_list']
    if not image_list:
        raise werkzeug.exceptions.BadRequest('image_list is a required field')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    paths = []
    images = []

    for line in image_list.readlines():
        line = line.strip()
        if not line:
            continue

        path = None
        # might contain a numerical label at the end
        match = re.match(r'(.*\S)\s+\d+$', line)
        if match:
            path = match.group(1)
        else:
            path = line

        try:
            image = utils.image.load_image(path)
            image = utils.image.resize_image(image,
                    job.load_model_task().crop_size, job.load_model_task().crop_size,
                    # TODO : get channels elegantly.
                    channels    = 3,
                    resize_mode = None,
                    )
            paths.append(path)
            images.append(image)
        except utils.errors.LoadImageError as e:
            print e

    if not len(images):
        raise werkzeug.exceptions.BadRequest(
                'Unable to load any images from the file')

    save_vis_file = False
    layers = None
    if 'save_visualizations' in flask.request.form and flask.request.form['save_visualizations']:
        save_vis_file = True

        # Check for specific layer
        if 'select_visualization_layer_bulk' in flask.request.form and flask.request.form['select_visualization_layer_bulk']:
            layers = flask.request.form['select_visualization_layer_bulk']
        else:
            layers = 'all'

        # Select save file type
        if 'save_type_mat_bulk' in flask.request.form and flask.request.form['save_type_mat_bulk']:
            save_file_type = 'mat'
        elif 'save_type_numpy_bulk' in flask.request.form and flask.request.form['save_type_numpy_bulk']:
            save_file_type = 'numpy'
        else:
            raise werkzeug.exceptions.BadRequest('No filetype selected. Expected .npy or .mat')
        
        # Obtain savefile path.
        if 'save_vis_file_location_bulk' in flask.request.form and flask.request.form['save_vis_file_location_bulk']:
            save_vis_file_location = flask.request.form['save_vis_file_location_bulk']

    if 'job_id' in flask.request.form and flask.request.form['job_id']:
        job_id = flask.request.form['job_id']
    elif 'job_id' in flask.request.args and flask.request.args['job_id']:
        job_id = flask.request.args['job_id']
    else:
        raise werkzeug.exceptions.BadRequest('job_id is a necessary parameter, not found.')

    labels, scores, visualizations = job.load_model_task().infer_many(images, snapshot_epoch=epoch, layers=layers)
    
    if scores is None:
        raise werkzeug.exceptions.BadRequest('An error occured while processing the images')

    # take top 5
    indices = (-scores).argsort()[:, :5]

    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # `i` is a category in labels and also an index into scores
            result.append((labels[i], round(100.0*scores[image_index, i],2)))
        classifications.append(result)

    layer_data = {}
    for image_vis in visualizations:
        for layer in image_vis:
            for ele in layer:
                if ele=='image_html':
                    continue
                if layer['name'] in layer_data:
                    if ele in layer_data[layer['name']]:
                        layer_data[layer['name']][ele].append(layer[ele])
                    else:
                        layer_data[layer['name']][ele] = [layer[ele]]
                else:
                    layer_data[layer['name']] = {}
                    layer_data[layer['name']][ele] = [layer[ele]]

    if save_vis_file:
        if save_file_type == 'numpy':
            try:
                joined_vis = layer_data
                np.array(joined_vis).dump(open(save_vis_file_location+'/visualization_'+job_id+'.npy', 'wb'))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as Numpy array')
        elif save_file_type == 'mat':
            try:
                joined_vis = layer_data
                scipy.io.savemat(save_vis_file_location+'/visualization_'+job_id+'.mat', {'visualizations':joined_vis})
            except IOError as e:
                raise werkzeug.exceptions.BadRequest('I/O error{%s}: %s'% (e.errno, e.strerror))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as .mat file')
        else:
            raise werkzeug.exceptions.BadRequest('Invalid filetype for visualization data saving')

    if request_wants_json():
        if 'save_visualizations' in flask.request.form and flask.request.form['save_visualizations']:
            joined_vis = layer_data
            joined_class = dict(zip(paths, classifications))
            return flask.jsonify({'classifications': joined_class, 'visualizations': joined_vis})
        else:
            joined = dict(zip(paths, classifications))
            return flask.jsonify({'classifications': joined})
    else:
        return flask.render_template('models/images/extraction/classify_many.html',
                paths=paths,
                classifications=classifications,
                )

@app.route(NAMESPACE + '/top_n', methods=['POST'])
@autodoc('models')
def extraction_model_top_n():
    """
    Classify many images and show the top N images per category by confidence
    """
    job = job_from_request()

    image_list = flask.request.files.get['image_list']
    if not image_list:
        raise werkzeug.exceptions.BadRequest('File upload not found')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])
    if 'top_n' in flask.request.form and flask.request.form['top_n'].strip():
        top_n = int(flask.request.form['top_n'])
    else:
        top_n = 9
    if 'num_test_images' in flask.request.form and flask.request.form['num_test_images'].strip():
        num_images = int(flask.request.form['num_test_images'])
    else:
        num_images = None

    paths = []
    for line in image_list.readlines():
        line = line.strip()
        if not line:
            continue

        path = None
        # might contain a numerical label at the end
        match = re.match(r'(.*\S)\s+\d+$', line)
        if match:
            path = match.group(1)
        else:
            path = line
        paths.append(path)
    random.shuffle(paths)

    images = []
    for path in paths:
        try:
            image = utils.image.load_image(path)
            image = utils.image.resize_image(image,
                    job.load_model_task().crop_size, job.load_model_task().crop_size,
                    #TODO : get channels elegantly.
                    channels    = 3,
                    resize_mode = None,
                    )
            images.append(image)
            if num_images and len(images) >= num_images:
                break
        except utils.errors.LoadImageError as e:
            print e

    if not len(images):
        raise werkzeug.exceptions.BadRequest(
                'Unable to load any images from the file')

    labels, scores = job.load_model_task().infer_many(images, snapshot_epoch=epoch)
    if scores is None:
        raise RuntimeError('An error occured while processing the images')

    indices = (-scores).argsort(axis=0)[:top_n]
    results = []
    for i in xrange(indices.shape[1]):
        result_images = []
        for j in xrange(top_n):
            result_images.append(images[indices[j][i]])
        results.append((
                labels[i],
                utils.image.embed_image_html(
                    utils.image.vis_square(np.array(result_images))
                    )
                ))

    return flask.render_template('models/images/extraction/top_n.html',
            job=job,
            results=results,
            )

