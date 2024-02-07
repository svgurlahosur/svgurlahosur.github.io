function selectOperation(){
    const operationType = document.getElementById('operationType').value;
    const convolutionOptionsDiv = document.getElementById('convolutionOptions');
    const poolingOptionsDiv = document.getElementById('poolingOptions');
    if (operationType === 'convolution') {
        convolutionOptionsDiv.style.display = 'block';
        poolingOptionsDiv.style.display = 'none';
    } else {
        convolutionOptionsDiv.style.display = 'none';
        poolingOptionsDiv.style.display = 'block';
    }
}
function calculateFeatureMap() {
    const inputSize = parseInt(document.getElementById('inputSize').value);
    const operationType = document.getElementById('operationType').value;
    let result = '';
    let inputSizeField = document.getElementById('inputSize');
    let numChannelsField = document.getElementById('numChannels');

    if (operationType === 'convolution') {
        const filterSize = parseInt(document.getElementById('filterSizeValue').value);
        const stride = parseInt(document.getElementById('strideValue').value);
        const padding = parseInt(document.getElementById('paddingValue').value);
        const numFilters = parseInt(document.getElementById('numFiltersValue').value);

        let featureMapHeight, featureMapWidth;

        featureMapHeight = Math.floor((inputSize + 2 * padding - filterSize) / stride) + 1;
        
        if ((inputSize - filterSize + 2 * padding) % stride === 0)
            featureMapWidth = Math.floor((inputSize - filterSize + 2 * padding) / stride) + 1;
        else
            featureMapWidth = Math.floor((inputSize - filterSize + 2 * padding) / stride) + 1;

        if (featureMapWidth < 1) featureMapWidth = 1; // Ensure minimum width is 1

        result = `Feature Map Size: [${featureMapHeight} * ${featureMapWidth} * ${numFilters}]`;

        const regex = /\[(\d+) \* (\d+) \* (\d+)\]/;
        const match = result.match(regex);
        if (match && match.length === 4) {
            inputSizeField.value = parseInt(match[1]); // Update Input Data Size
            numChannelsField.value = parseInt(match[3]); // Update Number of Channels
        }
    } else {
        const poolingFilterSize = parseInt(document.getElementById('poolingFilterSize').value);
        const numChannels = parseInt(document.getElementById('numChannels').value);
        const featureMapSize = Math.floor(inputSize / poolingFilterSize);
        result = `Feature Map Size: [${featureMapSize} * ${featureMapSize} * ${numChannels}]`;
        const regex = /\[(\d+) \* (\d+) \* (\d+)\]/;
        const match = result.match(regex);
        if (match && match.length === 4) {
            inputSizeField.value = parseInt(match[1]); // Update Input Data Size
            numChannelsField.value = parseInt(match[3]); // Update Number of Channels
        }
    }
    const outputDiv = document.getElementById('output');
    outputDiv.innerHTML = `<strong>${result}</strong>`;
}