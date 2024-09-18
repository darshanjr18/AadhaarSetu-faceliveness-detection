const video = document.createElement('video');
const cameraPreview = document.getElementById('cameraPreview');
const statusMessage = document.getElementById('statusMessage');
const aadhaarForm = document.getElementById('aadhaarForm');
const aadhaarNumberInput = document.getElementById('aadhaarNumber');
const modelPath = '/path/to/tfjs_model/model.json'; // Update with your model path

let model;



// Example function to show verification results
function showVerificationResult(isLive) {
    const resultElement = document.getElementById('verificationResult');
    const formElement = document.getElementById('aadhaarForm');

    if (isLive) {
        resultElement.innerHTML = "<p style='color: green;'>Face liveness detected. You can proceed further.</p>";
        formElement.style.display = "block";
    } else {
        resultElement.innerHTML = "<p style='color: red;'>Spoofing detected. You cannot proceed further.</p>";
        formElement.style.display = "none";
    }
}

// Function to start the camera
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });
    video.srcObject = stream;
    video.autoplay = true;
    video.width = cameraPreview.clientWidth;
    video.height = cameraPreview.clientHeight;
    cameraPreview.appendChild(video);
}

// Function to load TensorFlow.js model
async function loadModel() {
    model = await tf.loadGraphModel(modelPath);
    statusMessage.textContent = 'Model loaded. Detecting face liveness...';
}

// Function to detect face liveness
async function detectFaceLiveness() {
    const img = tf.browser.fromPixels(video).resizeNearestNeighbor([32, 32]).toFloat();
    const prediction = model.predict(img.expandDims(0));
    const isReal = prediction.dataSync()[0] > 0.5;
    return isReal;
}

// Function to start liveness detection
async function startLivenessDetection() {
    await setupCamera();
    await loadModel();
    
    try {
        const isReal = await detectFaceLiveness();
        statusMessage.textContent = isReal ? 'Liveness detection successful' : 'Liveness detection failed';
        cameraPreview.style.display = 'none';
        aadhaarForm.style.display = 'block'; // Show Aadhaar form after liveness detection
    } catch (error) {
        statusMessage.textContent = `Error: ${error.message}`;
    }
}

// Aadhaar form submission handler
aadhaarForm.addEventListener('submit', function(event) {
    event.preventDefault();
    const aadhaarNumber = aadhaarNumberInput.value;
    if (aadhaarNumber.length === 12) {
        statusMessage.textContent = 'Aadhaar verification in progress...';
        // Simulate Aadhaar verification (integrate with Aadhaar verification API here)
        setTimeout(() => {
            statusMessage.textContent = 'Aadhaar verified successfully!';
        }, 2000);
    } else {
        statusMessage.textContent = 'Invalid Aadhaar number';
    }
});

// Start the liveness detection when the page loads
window.onload = startLivenessDetection;




