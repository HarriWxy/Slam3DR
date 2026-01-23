import cv2
import threading
import multiprocessing as mp
import time
from http import server
import socketserver

class _MJPEGHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/feed", "/latest.jpg"):
            self.send_response(404)
            self.end_headers()
            return

        proxy = self.server.proxy_ref
        if self.path == "/latest.jpg":
            with proxy.lock:
                data = proxy.latest_jpeg
            if not data:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # /feed
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while not proxy.stop_event.is_set():
            with proxy.lock:
                data = proxy.latest_jpeg
            if data:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
            time.sleep(0.03)

class _ThreadingHTTPServer(socketserver.ThreadingMixIn, server.HTTPServer):
    daemon_threads = False

class MJPEGProxy:
    def __init__(self, src_url, host="127.0.0.1", port=8098, width=640, height=480):
        self.src_url = src_url
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.stop_event = threading.Event()
        self._server = None
        self._capture_thread = None
        self._server_thread = None

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.src_url)
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.width and self.height:
                frame = cv2.resize(frame, (self.width, self.height),interpolation=cv2.INTER_CUBIC)
            ok, jpg = cv2.imencode(".jpg", frame)
            if ok:
                with self.lock:
                    self.latest_jpeg = jpg.tobytes()
        cap.release()

    def start(self):
        if self._server is not None:
            return
        self.stop_event.clear()
        self._server = _ThreadingHTTPServer((self.host, self.port), _MJPEGHandler)
        self._server.proxy_ref = self
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=False)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=False)
        self._capture_thread.start()
        self._server_thread.start()



    def stop(self):
        self.stop_event.set()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        self._server = None



def test_run(src_url, width=640, height=480):
    cap = cv2.VideoCapture(src_url)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        if width and height:
            frame = cv2.resize(frame, (width, height))
        ok, jpg = cv2.imencode(".jpg", frame)
        if ok:
            latest_jpeg = jpg.tobytes()

# test
if __name__ == "__main__":
    src_url = "http://192.168.1.111:4747/video"
    _mjpeg_proxy = MJPEGProxy(src_url, host="127.0.0.1", port=8098, width=640, height=480)
    _mjpeg_proxy.start()
    # width = 640
    # height = 480
    # a = threading.Thread(target=test_run, args=(src_url, width, height), daemon=False)
    # a.start()
    # a.join()

    # cap.release()
