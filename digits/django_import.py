import os, sys
# If your project is in '/home/user/mysite/polls', you have to put sys.path.extend(['/home/user/mysite/'])
sys.path.extend(['/home/pydev/Documents/cloudcv17',
'/home/pydev/Documents/cloudcv17/env/lib/python2.7',
'/home/pydev/Documents/cloudcv17/env/lib/python2.7/plat-x86_64-linux-gnu', 
'/home/pydev/Documents/cloudcv17/env/lib/python2.7/lib-tk', 
'/home/pydev/Documents/cloudcv17/env/lib/python2.7/lib-old', 
'/home/pydev/Documents/cloudcv17/env/lib/python2.7/lib-dynload', 
'/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', 
'/usr/lib/python2.7/lib-tk', '/home/pydev/Documents/cloudcv17/env/local/lib/python2.7/site-packages', 
'/home/pydev/Documents/cloudcv17/env/lib/python2.7/site-packages']) 

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cloudcv17.settings")
from django.conf import settings
from redis_sessions_fork.session import SessionStore
from webapp import socketio

import os
import time
import subprocess
from threading import Thread
from flask import Flask, render_template, session, request
from flask.ext.socketio import SocketIO, emit, join_room, leave_room, \
    close_room, disconnect
# print "ok"
# from webapp import socketio
# print "ok"

thread = None

import commands

def get_download_status(source_size, dest_path):
    """
        Method for calculating the percentage of completion
        and then sending a respinse to page using socketIO.
        This process is repeating untill the copy is not 
        done completely. 
    """
    # dest_size = int(commands.getoutput('du -s '+dest_path).split()[0])
    source_size = int(source_size)
    dest_size = 1
    while(dest_size <= source_size):
        dest_size = int(commands.getoutput('du -s '+dest_path).split()[0])
        time.sleep(0.01)
        percent = (dest_size*100/source_size)
        # dest_size = get_size(dest_path)
        # dest_size+=1
        socketio.emit("my progressbar",
            {'percent':percent},namespace='/')

