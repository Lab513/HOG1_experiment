'''
Utilities for the interface
'''
import os, subprocess, json
import shutil as sh
opd, opb, opj = os.path.dirname, os.path.basename, os.path.join
from sys import platform as _platform
import threading, webbrowser

def find_platform():
    '''
    Find which platform is currently used
    '''
    print('platform is ',_platform)
    if _platform == "linux" or _platform == "linux2":
       platf = 'lin'
    elif _platform == "darwin":
       platf = 'mac'
    elif _platform == "win32":
       platf = 'win'
    return platf

def try_nb(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def clean_dir(dir):
    '''
    Clear folder dir
    '''
    try:
        for f in glob.glob(opj(dir, '*.*')):
            os.remove(f)
    except:
        print(f"can't clean {dir}")

def open_folder(path):
    '''
    Open folder
    '''
    if platf == 'win':
        os.startfile(path)
    elif platf == 'mac':
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def rm_make_upload(config):
    '''
    '''
    upload = config['UPLOADED_PATH']
    try:
        sh.rmtree(upload)                        # Reinitializes the upload folder
    except:
        print(f"cannot find {upload}")
    os.mkdir(upload)

def init(config):
    '''
    Prepare interface.
    '''
    rm_make_upload(config)

def find_chrome_path(platf):
    '''
    '''
    # MacOS
    if platf == 'mac':
        chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Linux
    elif platf == 'lin':
        chrome_path = '/usr/bin/google-chrome %s'
    else:
        chrome_path = False
    return chrome_path

def launch_browser(port, host, platf, debug=[]):
    '''
    Launch Chrome navigator
    '''
    chrome_path = find_chrome_path(platf)
    url = f"http://{host}:{port}" #
    if platf != 'win':
        b = webbrowser.get(chrome_path)
        threading.Timer(1.25, lambda: b.open_new(url)).start()    # open a page in the browser.
    else:
        try:
            if 0 in debug:
                print('using first path for Chrome')
            subprocess.Popen(f'"C:\Program Files\Google\Chrome\Application\chrome.exe" {url}')
        except:
            if 0 in debug:
                print('using second path for Chrome')
            subprocess.Popen(f'"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" {url}')
