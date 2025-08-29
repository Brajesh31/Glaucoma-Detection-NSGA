document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const uploadedImage = document.getElementById('uploaded-image');
    const predictionText = document.getElementById('prediction-text');
    const confidenceLevel = document.getElementById('confidence-level');
    const resultMessage = document.getElementById('result-message');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultContent = document.getElementById('result-content');
    const gradCamImage = document.getElementById('grad-cam-image');
    const gradCamArea = document.getElementById('grad-cam-area');

    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            // Reset UI state
            resultContent.classList.add('d-none');
            gradCamArea.classList.add('d-none');
            loadingSpinner.classList.remove('d-none');

            const formData = new FormData();
            const file = imageInput.files[0];
            formData.append('image', file);

            try {
                // Display the uploaded image
                const reader = new FileReader();
                reader.onload = function(event) {
                    uploadedImage.src = event.target.result;
                    uploadedImage.classList.remove('d-none');
                };
                reader.readAsDataURL(file);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    const prediction = result.prediction;
                    const confidence = (result.confidence * 100).toFixed(2);
                    const gradCamImageBase64 = result.grad_cam_image;

                    predictionText.textContent = prediction;
                    confidenceLevel.textContent = confidence;

                    if (prediction === "Glaucoma") {
                        resultMessage.innerHTML = `<span class="text-danger fw-bold"><i class="fas fa-exclamation-triangle"></i> Alert:</span> This image shows signs of Glaucoma. Please consult a doctor.`;
                    } else {
                        resultMessage.innerHTML = `<span class="text-success fw-bold"><i class="fas fa-check-circle"></i> Result:</span> Your eyes appear healthy.`;
                    }

                    if (gradCamImageBase64) {
                        gradCamImage.src = `data:image/jpeg;base64,${gradCamImageBase64}`;
                        gradCamArea.classList.remove('d-none');
                    } else {
                        gradCamArea.classList.add('d-none');
                    }

                    resultContent.classList.remove('d-none');
                } else {
                    resultMessage.innerHTML = `<span class="text-danger fw-bold"><i class="fas fa-times-circle"></i> Error:</span> ${result.error || 'There is a server error.'}`;
                    resultContent.classList.remove('d-none');
                }
            } catch (error) {
                resultMessage.innerHTML = `<span class="text-danger fw-bold"><i class="fas fa-times-circle"></i> Network Error:</span> Unable to connect to the server.`;
                resultContent.classList.remove('d-none');
                console.error('Error:', error);
            } finally {
                loadingSpinner.classList.add('d-none');
            }
        });
    }
});
