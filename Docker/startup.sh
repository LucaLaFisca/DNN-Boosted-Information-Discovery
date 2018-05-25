DEBUG=false
if ${DEBUG}; then
python3.5 server_CTPN.py &
python3.5 server_SSD.py &
python3.5 server_MASK_RCNN.py &
python3.5 server_main.py
fi
python3.5 server_main.py