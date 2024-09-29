import distributionParams from './assets/distributions.json' with { type: "json" };

const distributionSelect = document.getElementById('distribution-select');
const parametersFormContainer = document.getElementById('parameters-form');
const parametersForm = document.getElementById('distribution-form');
const computeFormContainer = document.getElementById('compute-form-container');
const computeForm = document.getElementById('compute-form');
const resultsContainer = document.getElementById('results');
const resultSpan = document.getElementById('computed-result');
const chartContainer = document.getElementById('chart-container');
const chartCanvas = document.getElementById('chart');
const distributionDescription = document.getElementById('distribution-description');
const distributionDescText = document.getElementById('distribution-desc-text');
const distributionProperties = document.getElementById('distribution-properties');
const distributionMean = document.getElementById('distribution-mean');
const distributionVariance = document.getElementById('distribution-variance');
let chart = null;  // Reference to the Chart.js instance

// Populate the select dropdown with distribution names
const populateDistributions = () => {
    for (const [key, value] of Object.entries(distributionParams)) {
        const option = document.createElement('option');
        option.value = key;
        option.text = key;
        distributionSelect.appendChild(option);
    }
};

// Generate form inputs based on the selected distribution
const generateForm = (distribution) => {
    parametersForm.innerHTML = '';  // Clear previous form

    const params = distributionParams[distribution].parameters;
    for (const [param, details] of Object.entries(params)) {
        const div = document.createElement('div');
        div.classList.add('form-group');  // Adding a class to the div for styling

        const label = document.createElement('label');
        label.innerHTML = `${details.description}: `;

        const input = document.createElement('input');
        input.type = details.type === 'integer' ? 'number' : 'text';
        input.name = param;
        input.placeholder = details.range ? `${details.range[0]} to ${details.range[1]}` : '';

        // Append label and input to the div
        div.appendChild(label);
        div.appendChild(input);

        // Append the div to the form
        parametersForm.appendChild(div);
    }

    const submitBtn = document.createElement('button');
    submitBtn.id = "generateChartButton";
    submitBtn.type = 'submit';
    submitBtn.textContent = 'Generate Chart';

    parametersForm.appendChild(submitBtn);

    parametersFormContainer.style.display = 'block';
    computeFormContainer.style.display = 'block';
};

// Function to handle chart type (continuous or histogram) and generate the chart
parametersForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const distribution = distributionSelect.value;
    const formData = new FormData(parametersForm);
    const parameters = {};
    let hasEmptyFields = false;

    for (const [key, value] of formData.entries()) {
        if (!value) {
            alert(`Please fill in the ${key} field.`);
            hasEmptyFields = true;
            break;
        }
        parameters[key] = parseFloat(value);
    }

    if (hasEmptyFields) return;  // Stop if there are empty fields

    // Call eel function to compute the distribution and generate the data
    eel.compute_distribution(distribution, parameters)((result) => {
        const { xs, ys, variableType } = result;

        // Determine chart type: 'line' for continuous, 'bar' for discrete
        const chartType = variableType === 'continuous' ? 'line' : 'bar';

        // Display chart
        chartContainer.style.display = 'block';
        plotChart(xs, ys, chartType);
    });
});

// Function to plot chart using Chart.js
const plotChart = (xs, ys, chartType) => {
    if (chart) {
        chart.destroy();  // Destroy existing chart if present
    }

    chart = new Chart(chartCanvas, {
        type: chartType,  // 'line' or 'bar'
        data: {
            labels: xs,
            datasets: [{
                label: 'Probability',
                data: ys,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: chartType === 'bar' ? 'rgba(75, 192, 192, 0.2)' : 'rgba(0, 0, 0, 0)',
                borderWidth: 1,
                radius: 0,
                fill: chartType === 'line'
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'X values'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'PDF'
                    }
                }
            }
        }
    });
};

// Handle computation of a specific probability value
computeForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const distribution = distributionSelect.value;
    const computeValue = document.getElementById('computeValue').value;
    const formData = new FormData(parametersForm);
    const parameters = {};
    let hasEmptyFields = false;

    for (const [key, value] of formData.entries()) {
        if (!value) {
            alert(`Please fill in the ${key} field.`);
            hasEmptyFields = true;
            break;
        }
        parameters[key] = parseFloat(value);
    }

    if (!computeValue) {
        alert('Please provide a value to compute.');
        return;
    }

    if (hasEmptyFields) return;  // Stop if there are empty fields

    // Call eel function to compute the probability for the given value
    eel.compute_probability(distribution, parameters, parseFloat(computeValue))((computedValue) => {
        resultSpan.textContent = ` ${computedValue}`;
        resultsContainer.style.display = 'inline';
    });
});

// Event listener for dropdown selection change
distributionSelect.addEventListener('change', (e) => {
    const selectedDistribution = e.target.value;
    
    if (selectedDistribution === "") {
        parametersFormContainer.style.display = 'none';
        computeFormContainer.style.display = 'none';
        distributionDescription.style.display = 'none';
        resultsContainer.style.display = 'none';
    } else {
        generateForm(selectedDistribution);

        // Display the distribution description and properties
        const distributionInfo = distributionParams[selectedDistribution];
        distributionDescText.textContent = distributionInfo.description;
        distributionDescription.style.display = 'block';
        distributionDescription.classList.add("distribution-description");

        const properties = distributionInfo.properties;
        let meanText = `\\(${properties["mean"]}\\)`;
        let varianceText = `\\(${properties["variance"]}\\)`;

        distributionMean.innerHTML = meanText;  // Use innerHTML for LaTeX
        distributionVariance.innerHTML = varianceText;  // Use innerHTML for LaTeX        
        // Update MathJax to render the LaTeX
        MathJax.typesetPromise();
    }
});

// Initialize the form with distribution options
populateDistributions();
