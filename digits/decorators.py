from django_import import *
from functools import wraps
from flask import g, request, redirect, url_for
from digits.base_workspace import get_user_from_session, get_workspace_details
from organizations.models import OrganizationUser

import flask
import redis
redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

def login_required(f):
	@wraps(f)
	def decorated_function(*args, **kwargs):
		djsession_id = request.cookies.get("sessionid")
		if djsession_id is None:
			return redirect("http://localhost:8000/accounts/login", code = 302)

		key = djsession_id
		session_store = SessionStore(key)
		auth = session_store.load()

		if not auth:
			return redirect("http://localhost:8000/accounts/login", code = 302)

		g.user_id = str(auth.get("sessionid"))
		return f(*args, **kwargs)
	return decorated_function

def access_required(f):
	@wraps(f)
	def decorated_function(*args, **kwargs):
		workspace = get_workspace_details(flask.request.url)
		key = request.cookies.get("sessionid")
		user = get_user_from_session(key)
		if not OrganizationUser.objects.filter(user = user, organization__name = workspace['workspace_name']):
			return "<h2>You are not Authorised to work on this Workspace</h2>" 
		return f(*args, **kwargs)
	return decorated_function

