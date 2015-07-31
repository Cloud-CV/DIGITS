import os, sys
# If your project is in '/home/user/mysite/polls', you have to put sys.path.extend(['/home/user/mysite/'])
sys.path.extend(['/home/pydev/Documents/Workspaces', 
	'/home/pydev/Documents/Workspaces/env/lib/python2.7',
	'/home/pydev/Documents/Workspaces/env/lib/python2.7/plat-x86_64-linux-gnu', 
	'/home/pydev/Documents/Workspaces/env/lib/python2.7/lib-tk', 
	'/home/pydev/Documents/Workspaces/env/lib/python2.7/lib-old', 
	'/home/pydev/Documents/Workspaces/env/lib/python2.7/lib-dynload', 
	'/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', 
	'/usr/lib/python2.7/lib-tk', '/home/pydev/Documents/Workspaces/env/local/lib/python2.7/site-packages', 
	'/home/pydev/Documents/Workspaces/env/lib/python2.7/site-packages']) 

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Workspaces.settings")
from django.conf import settings
from redis_sessions_fork.session import SessionStore

