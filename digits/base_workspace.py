from django_import import *
from digitsdb.models import Job as WorkspaceJob
from organizations.models import OrganizationUser, Organization
from urlparse import urlparse, parse_qs
from django.contrib.auth import SESSION_KEY, BACKEND_SESSION_KEY, load_backend
from digitsdb.views import get_download_status
import base64

def get_workspace_details(url):
	result = {}
	# print parse_qs(urlparse(str(url)).query, keep_blank_values=True)
	workspace_id = parse_qs(urlparse(str(url)).query, keep_blank_values=True)["workspace"][0].encode('utf-8')
	result['workspace_id'] = base64.b64decode(workspace_id)
	result['workspace_hash'] = str(workspace_id)
	result['workspace_name'] = Organization.objects.get(pk = str(result['workspace_id'])).name.encode('utf-8')
	return result

def delete_job_from_workspace(job_id, workspace):
	workspace = Organization.objects.get(id = workspace['workspace_id'])
	WorkspaceJob.objects.filter(job_id = job_id, workspace = workspace).delete()

def get_user_from_session(session_key):
	session_engine = __import__(settings.SESSION_ENGINE, {}, {}, [''])
	session_wrapper = session_engine.SessionStore(session_key)
	session = session_wrapper.load()
	user_id = session.get(SESSION_KEY)
	backend_id = session.get(BACKEND_SESSION_KEY)
	if user_id and backend_id:
		auth_backend = load_backend(backend_id)
		user = auth_backend.get_user(user_id)
		if user:
			return user
	return AnonymousUser()

