function uploadVideo() {
    var fileInput = document.getElementById('videoInput');
    var file = fileInput.files[0];
    var formData = new FormData();
    formData.append('video', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        var videoContainer = document.getElementById('videoContainer');
        // Assuming data contains the path returned from the server
        var videoURL = data.replace(/\\/g,"/" ); // Modify this if the data format is different
        console.log("/static/"+videoURL)
        // videoContainer.innerHTML = `<video controls><source src="/static/${videoURL}" type="video/*"></video>`;
        videoContainer.innerHTML = `<div class='text-success mx-auto mt-2 row justify-content-center'>Video uploaded successfully!</div>`

        // Enable predict button after video upload
        var predictBtn = document.getElementById('predictbtn');
        predictBtn.removeAttribute('disabled');
        
        // Add click event listener to predict button
        predictBtn.addEventListener('click', function() {
            var alertDiv = document.getElementById('output');
            alertDiv.innerHTML = 'Processing...'; 
            // Send request to predict route
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Assuming the response contains predictions
                var predictions = data.predictions;
                var outputDiv = document.getElementById('output');
                outputDiv.innerHTML = ''; // Clear previous predictions
                
                // Display predictions
                predictions.forEach(prediction => {
                    var p = document.createElement('span');
                    p.textContent = prediction;
                    outputDiv.appendChild(p);
                });
            })
            .catch(error => console.error('Error predicting speech:', error));
        });
    })
    .catch(error => console.error('Error:', error));
}

function myFunction() {
    // Get the text field
    var copyText = document.getElementById("output");
  
    // Create a temporary input element to hold the text
  var tempInput = document.createElement("input");
  tempInput.setAttribute("type", "text");
  tempInput.setAttribute("value", copyText.textContent); // Copy the text content of the span

  // Append the temporary input element to the document body
  document.body.appendChild(tempInput);

  // Select the text inside the temporary input
  tempInput.select();
  tempInput.setSelectionRange(0, 99999); // For mobile devices

  // Copy the selected text
  document.execCommand("copy");

  // Remove the temporary input element
  document.body.removeChild(tempInput);

  // Alert the copied text (optional)
  alert("Copied the text: " + copyText.textContent);
  }
