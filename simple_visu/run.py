#!/usr/bin/env python
# encoding: utf-8

"""

Simple interface

pip install flask-socketio
pip install eventlet
pip install eventlet==0.26 (for Windows)

python -m simple_visu.run

"""
from __future__ import print_function, division, absolute_import

import os
import yaml
import json
op = os.path
opd, opb, opj = op.dirname, op.basename, op.join
import shutil as sh
from colorama import Fore, Style                      # Color in the Terminal

import webbrowser
import threading

##
from matplotlib import pyplot as plt
import numpy as np
# Flask imports
from flask import Flask, render_template, request, redirect
from flask_socketio import SocketIO, emit
from simple_visu.modules.pages.define_all_pages import *
from simple_visu.modules.util_interf import *

platf = find_platform()

if platf == 'win':
    import gevent as server
    from gevent import monkey
    monkey.patch_all()
else:
    import eventlet as server
    server.monkey_patch()

Debug = True                                          # Debug Flask

app = Flask(__name__)
# upload directory from the Dropzone
app.config['UPLOADED_PATH'] = opj(os.getcwd(), 'simple_visu', 'upload')
print(f"######### app.config['UPLOADED_PATH'] "
      " is {app.config['UPLOADED_PATH']} !!!")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'F34TF$($e34D'
socketio = SocketIO(app)


@socketio.on('connect')
def test_connect():
    '''
    Websocket connection
    '''
    send_nb_pics()
    emit('response', 'Connected')
    server.sleep(0.05)


@app.route('/', methods=['GET', 'POST'])
def main_page(debug=1):
    '''
    '''
    platf = find_platform()
    print(f'platf is {platf}')
    dmp = define_main_page(platf)
    return render_template('index_folder.html', **dmp.__dict__)


def send_nb_pics():
    '''
    Sending the number of pictures
    '''
    with open('simple_visu/nb_pics.yaml') as f:
        nb_pics = yaml.load(f, Loader=yaml.FullLoader)
        print(f'nb_pics is {nb_pics}')
    emit('nb_pics', nb_pics)


@socketio.on('mess')
def receiving_mess(mess):
    '''
    Receiving a message
    '''
    print(f"mess is {mess}")
    emit('refresh', "")


@socketio.on('retrieve_title')
def retrieve_title():
    '''
    Searching for the title
    '''
    with open('jupyter_for_analysis/experim_title.yaml') as f_r:
        title = yaml.load(f_r, Loader=yaml.FullLoader)
    #print(f'The title of the experiment is {title}')
    emit('title', title)


@socketio.on('select_all_cells')
def select_all_cells(frm_pos):
    '''
    Select all the cells in the current frame
    '''
    reinit_cells('whatever')
    ldic_frm_pos = make_ldic_frm_obj(frm_pos, 'pos')      # frame and positions
    with open('jupyter_for_analysis/pos.yaml', 'w') as f:
        yaml.dump(ldic_frm_pos, f)                      # select all the cells

@socketio.on('reinit_cells')
def reinit_cells(mess):
    '''
    Empty pos.yaml file..
    '''
    with open('jupyter_for_analysis/pos.yaml', 'w') as f:
        yaml.dump([], f)                                      # reinit pos.yaml
    with open('jupyter_for_analysis/selected_area.yaml', 'w') as f:
        yaml.dump([], f)                            # reinit selected_area.yaml


def make_ldic_frm_obj(frm_obj, obj):
    '''
    list of dictionaries {frame, x, y}
    '''
    try:
        with open(f'jupyter_for_analysis/{obj}.yaml') as f:
            ldic_frm_obj = yaml.load(f, Loader=yaml.FullLoader)
    except:
        ldic_frm_obj = []
    if ldic_frm_obj and ldic_frm_obj != list:
        if type(ldic_frm_obj) == dict:
            ldic_frm_obj = [ldic_frm_obj]
    else:
        ldic_frm_obj = []
    frm_obj = json.loads(frm_obj)
    ldic_frm_obj += [frm_obj]

    return ldic_frm_obj


@socketio.on('mouse_pos')
def mousepos(frm_pos, debug=[0]):
    '''
    Mouse position
    '''
    if 0 in debug:
        print(f"Mouse position is {frm_pos}")
    ldic_frm_pos = make_ldic_frm_obj(frm_pos, 'pos')      # frame and positions
    with open('jupyter_for_analysis/pos.yaml', 'w') as f_w:
        yaml.dump(ldic_frm_pos, f_w)            # dump the list of dictionaries
        print(f'save pos {ldic_frm_pos}')

@socketio.on('select_area')
def select_area(frm_rect_coord, debug=[0]):
    '''
    Selected area
    '''
    if 0 in debug:
        print(f"Selected area is {frm_rect_coord}")
    # frame and rectangle coordinates
    ldic_frm_coord = make_ldic_frm_obj(frm_rect_coord, 'selected_area')
    with open('jupyter_for_analysis/selected_area.yaml', 'w') as f:
        yaml.dump(ldic_frm_coord, f)                       # dump dic_frm_coord
        print(f'save ldic_frm_coord {ldic_frm_coord}')

def shutdown_server():
    '''
    Quit the application
    called by method shutdown() (hereunder)
    '''
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown')
def shutdown():
    '''
    Shutting down the server.
    '''
    shutdown_server()

    return 'Server shutting down...'


def message_at_beginning(host, port):
    '''
    '''
    print(Fore.YELLOW + f"""
    ***************************************************************
    Launching the simple_visu server program !!!

    address: {host}:{port}

    Addons :

    pip install flask-socketio
    pip install gevent (Windows)
    pip install eventlet

    Change each time the port !!!
    perhaps using random port..

    """)

if __name__ == '__main__':
    init(app.config)                # clean last processings and upload folders

    port = 5975
    host = '0.0.0.0'                 # simple_visu.run
    print("host is ", host)
    message_at_beginning(host, port)
    print(Style.RESET_ALL)
    socketio.run(app, port=port, host=host)
