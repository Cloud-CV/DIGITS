from django_import import *
from digitsdb.models import Job as WorkspaceJob
from organizations.models import OrganizationUser, Organization
from urlparse import urlparse, parse_qs
import base64

def get_workspace_details(url):
	result = {}
	workspace_id = parse_qs(urlparse(str(url)).query, keep_blank_values=True)["workspace"][0].encode('utf-8')
	result['workspace_id'] = base64.b64decode(workspace_id)
	result['workspace_hash'] = str(workspace_id)
	result['workspace_name'] = Organization.objects.get(pk = str(result['workspace_id'])).name.encode('utf-8')
	return result

def delete_job_from_workspace(job_id, workspace):
	workspace = Organization.objects.get(id = workspace['workspace_id'])
	WorkspaceJob.objects.filter(job_id = job_id, workspace = workspace).delete()
