from __future__ import print_function

from flask import Flask, redirect, request, url_for, send_file
from redis import Redis
import re
import os
#from import_unimport_function import import_unimport, compute_img
from werkzeug.utils import secure_filename
from networks import network
import_state = network(0, 'None')

#8001 : SSD
#8002 : CTPN
#8003 : Mask_RCNN
#8004 : face_detector
#8005 : face_landmark_detection
#8006 : face_CV_network
#8007 : score
#8007 : score_360 (same port)
#8008 : retina_net (faces)
#8009 : retina_net (coco)
#8010 : research

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG'])

redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))
    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
        <p><input type=file name=file>
        <input type=submit value=Upload>
        </form>
        '''

def import_modification(new_state_port, new_state_name, new_state_address='/compute'):
    global import_state
    if import_state.port != new_state_port or import_state.address != new_state_address:
        if import_state.port != 0:
            import_state.stop_network()
        import_state.port = new_state_port
        import_state.name = new_state_name
        import_state.address = new_state_address
        import_state.start_network()

@app.route("/")
def main():
    html = "<!doctype html>" \
            "<title>Main page</title>" \
            "<meta charset=utf-8>" \
            '<a href="/upload-SSD/">Upload photo for SSD</a><br>' \
            '<a href="/upload-CTPN/">Upload photo for CTPN</a><br>' \
            '<a href="/upload-Mask_RCNN/">Upload photo for Mask_RCNN</a><br>' \
            '<a href="/upload-face_detector/">Upload photo for face_detector</a><br>' \
            '<a href="/upload-face_landmark/">Upload photo for face_landmark</a><br>' \
            '<a href="/upload-face_CV_network/">Upload photo for face_CV_network</a><br>' \
            '<a href="/upload-retina_net/">Upload photo for retina_net (faces)</a><br>' \
            '<a href="/upload-retina_net_coco/">Upload photo for retina_net (coco)</a><br>' \
            '<a href="/upload-score/">Upload photo for score</a><br>' \
            '<a href="/upload-score_360/">Upload 360 photo for score</a><br>' \
            '<a href="/upload-research/">Upload for CSV research</a><br>'
    return html

@app.route("/upload-SSD/", methods=['GET', 'POST'])
def redirect_SSD():
    import_modification(8001, 'SSD_network.py')
    return upload_file()

@app.route("/upload-CTPN/", methods=['GET', 'POST'])
def redirect_CTPN():
    import_modification(8002, 'CTPN_network.py')
    return upload_file()

@app.route("/upload-Mask_RCNN/", methods=['GET', 'POST'])
def redirect_Mask_RCNN():
    import_modification(8003, 'Mask_RCNN_network.py')
    return upload_file()

@app.route("/upload-face_detector/", methods=['GET', 'POST'])
def redirect_face_detector():
    import_modification(8004, 'face_detector_network.py')
    return upload_file()

@app.route("/upload-face_landmark/", methods=['GET', 'POST'])
def redirect_face_landmark():
    import_modification(8005, 'face_landmark_network.py')
    return upload_file()

@app.route("/upload-face_CV_network/", methods=['GET', 'POST'])
def redirect_upload_face_CV_network():
    import_modification(8006, 'face_CV_network.py')
    return upload_file()

@app.route("/upload-score/", methods=['GET', 'POST'])
def redirect_upload_score():
    import_modification(8007, 'score_network.py')
    return upload_file()

@app.route("/upload-score_360/", methods=['GET', 'POST'])
def redirect_upload_score_360():
    import_modification(8007, 'score_network.py', '/360')
    return upload_file()


@app.route("/upload-retina_net/", methods=['GET', 'POST'])
def redirect_upload_retina_net():
    import_modification(8008, 'retina_net_network.py')
    return upload_file()

@app.route("/upload-retina_net_coco/", methods=['GET', 'POST'])
def redirect_upload_retina_net_coco():
    import_modification(8009, 'retina_net_coco_network.py')
    return upload_file()

@app.route("/upload-research/", methods=['GET', 'POST'])
def redirect_upload_research():
    import_modification(8010, 'research_classifying.py')
    return upload_file()

@app.route('/upload/<filename>')
def uploaded_file(filename):
    global import_state
    filename = os.path.join(UPLOAD_FOLDER, filename)

    if import_state.port != 8010:
        output_path = import_state.compute_img(filename)
        print(output_path)
        return send_file(output_path, mimetype='image/gif')
    else:
        return import_state.compute_img(filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)