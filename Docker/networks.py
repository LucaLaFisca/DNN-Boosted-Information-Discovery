import os
import requests
import time

def wait_loading(url):
    while True:
        error = False
        time.sleep(1)
        try:
            r = requests.get(url)
        except requests.exceptions.RequestException:
            error = True
        #print('error')
        if error == False:
            if r.status_code == 200:
                #print('RUNNING')
                return

class network:
    def __init__(self, port, name, address='/compute'):
        self.port = port
        self.name = name
        self.address = address

    def start_network(self):
        os.system('python3.5 ' + self.name + ' &')
        url = 'http://localhost:' + str(self.port) + '/running'
        wait_loading(url)

    def stop_network(self):
        r = requests.get('http://localhost:' + str(self.port) + '/shutdown')
        print(self.name, r.status_code)

    def compute_img(self, filename):
        payload = {'key1': filename}
        r = requests.get('http://localhost:' + str(self.port) + self.address, params=payload)
        print(self.name, r.status_code, r.text)
        return r.text