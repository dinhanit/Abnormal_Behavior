const videoElement = document.getElementById('camera-feed');
const socket = new WebSocket('ws://localhost:8000/ws/camera');

socket.onmessage = (event) => {
    const frameBlob = event.data;
    const frameURL = URL.createObjectURL(frameBlob);
    videoElement.src = frameURL;
};
