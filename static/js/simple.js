const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");

const predictionText = document.getElementById("prediction");
const confidenceText = document.getElementById("confidence");
const heatmapWrap = document.getElementById("heatmapWrap");
const heatmapImg = document.getElementById("heatmapImg");
const reportLink = document.getElementById("reportLink");
const downloadOriginal = document.getElementById("downloadOriginal");

let selectedImage = null;

/* IMAGE PREVIEW */
imageInput.addEventListener("change", () => {
    selectedImage = imageInput.files[0];

    if (selectedImage) {
        preview.src = URL.createObjectURL(selectedImage);
        preview.style.display = "block";
    }
});

/* PREDICTION */
predictBtn.addEventListener("click", async () => {

    if (!selectedImage) {
        alert("Please upload an image first!");
        return;
    }

    loader.style.display = "block";
    predictionText.innerText = "";
    confidenceText.innerText = "";

    const formData = new FormData();
    // our backend expects form field named 'file' (not 'image')
    formData.append("file", selectedImage);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const txt = await response.text();
            throw new Error(txt || 'server error');
        }

        const data = await response.json();

        // backend returns 'predicted_class' and 'confidence' (0..1)
        const cls = data.predicted_class || data.class || 'Unknown';
        const conf = (data.confidence !== undefined ? data.confidence : (data.confidence_percent || 0));
        const confPct = (conf > 1 ? conf : conf * 100).toFixed(1);

        predictionText.innerText = "Disease: " + cls;
        confidenceText.innerText = "Confidence: " + confPct + "%";

        // show heatmap and report if provided
        if (data.heatmap_url) {
            heatmapImg.src = data.heatmap_url;
            heatmapWrap.style.display = 'block';
        } else {
            heatmapWrap.style.display = 'none';
        }
        if (data.report_url) {
            reportLink.href = data.report_url;
            reportLink.style.display = 'inline-block';
        } else {
            reportLink.style.display = 'none';
        }
        if (data.prediction_id) {
            downloadOriginal.href = '/static/uploads/' + (data.original_filename || '').replace(/ /g, '%20') || '#';
            downloadOriginal.style.display = 'inline-block';
        } else {
            downloadOriginal.style.display = 'none';
        }

    } catch (error) {
        alert("Error connecting to server: " + (error.message || error));
    } finally {
        loader.style.display = "none";
    }
});
