# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import flask
import werkzeug.exceptions

from digits.webapp import app, scheduler, autodoc
from digits.utils.routing import request_wants_json
import images.views
import images as dataset_images
from digits.base_workspace import *
from digits.decorators import login_required, access_required

NAMESPACE = '/datasets/'

@app.route(NAMESPACE + '<job_id>.json', methods=['GET'])
@app.route(NAMESPACE + '<job_id>', methods=['GET'])
@autodoc(['datasets', 'api'])
@login_required
@access_required
def datasets_show(job_id):
    """
    Show a DatasetJob

    Returns JSON when requested:
        {id, name, directory, status}
    """
    workspace = get_workspace_details(flask.request.url)
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, dataset_images.ImageClassificationDatasetJob):
            return dataset_images.classification.views.show(job, workspace)
        elif isinstance(job, dataset_images.FeatureExtractionDatasetJob):
            return dataset_images.extraction.views.show(job, workspace)
        else:
            raise werkzeug.exceptions.BadRequest('Invalid job type')

@app.route(NAMESPACE + 'summary', methods=['GET'])
@autodoc('datasets')
@login_required
@access_required
def dataset_summary():
    """
    Return a short HTML summary of a DatasetJob
    """
    workspace = get_workspace_details(flask.request.url)
    job = scheduler.get_job(flask.request.args['job_id'])
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    return flask.render_template('datasets/summary.html', dataset=job, workspace = workspace)

