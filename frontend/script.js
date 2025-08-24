function predictResult() {
    // Require all dropdown and number inputs to be filled
    const inputs = document.querySelectorAll("select, input[type='number']");
    for (let input of inputs) {
        if (input.value === "") {
            alert("Please fill all fields before predicting.");
            return;
        }
    }
    // Prepare form data
    const data = {
        continent: document.getElementById('continent').value,
        education_of_employee: document.getElementById('education_of_employee').value,
        has_job_experience: document.getElementById('has_job_experience').value,
        requires_job_training: document.getElementById('requires_job_training').value,
        region_of_employment: document.getElementById('region_of_employment').value,
        unit_of_wage: document.getElementById('unit_of_wage').value,
        full_time_position: document.getElementById('full_time_position').value,
        no_of_employees: Number(document.getElementById('no_of_employees').value),
        yr_of_estab: Number(document.getElementById('yr_of_estab').value),
        prevailing_wage: Number(document.getElementById('prevailing_wage').value)
    };
    // Call backend
    fetch("http://192.168.29.203:5000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {
        if (result.error) {
            alert("Error: " + result.error);
        } else {
            const box = document.getElementById("resultBox");
            box.innerText = "🎯 Predicted Case Status: " + result.prediction;
            box.style.display = "block";
        }
    })
    .catch(err => alert("Request failed: " + err));
}
