function createInputTable() {
    const inputSize = parseInt(document.getElementById('inputSize').value);
    const inputTableDiv = document.getElementById('inputTable');
    inputTableDiv.innerHTML = '<strong>Input Data:</strong><br>';
    const table = document.createElement('table');

    for (let i = 0; i < inputSize; i++) {
        const row = document.createElement('tr');
        for (let j = 0; j < inputSize; j++) {
            const cell = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'number';
            input.value = "1"; // Set default value to 1
            input.min = "1"; // Set minimum value to 1
            input.max = "10"; // Set maximum value to 10
            cell.appendChild(input);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
    inputTableDiv.appendChild(table);

    // Show pooling inputs after creating the table
    document.getElementById('poolingInputs').style.display = 'block';
}

function performMaxPooling() {
    const inputSize = parseInt(document.getElementById('inputSize').value);
    const poolSize = parseInt(document.getElementById('poolSize').value);

    const inputData = getInputData(inputSize);
    const pooledData = maxPooling(inputData, poolSize);

    displayPooledData(pooledData);
}

function performAveragePooling() {
    const inputSize = parseInt(document.getElementById('inputSize').value);
    const poolSize = parseInt(document.getElementById('poolSize').value);

    const inputData = getInputData(inputSize);
    const pooledData = averagePooling(inputData, poolSize);

    displayPooledData(pooledData);
}

function getInputData(size) {
    const data = new Array(size);
    const inputTable = document.querySelectorAll('#inputTable input');

    let index = 0;
    for (let i = 0; i < size; i++) {
        data[i] = new Array(size);
        for (let j = 0; j < size; j++) {
            const inputValue = parseInt(inputTable[index].value);
            data[i][j] = inputValue;
            index++;
        }
    }
    return data;
}

function maxPooling(inputData, poolSize) {
    const size = inputData.length;
    const pooledSize = Math.floor(size / poolSize);

    const pooledData = new Array(pooledSize);
    for (let i = 0; i < pooledSize; i++) {
        pooledData[i] = new Array(pooledSize);
        for (let j = 0; j < pooledSize; j++) {
            let max = 0;
            for (let m = 0; m < poolSize; m++) {
                for (let n = 0; n < poolSize; n++) {
                    const value = inputData[i * poolSize + m][j * poolSize + n];
                    if (value > max) {
                        max = value;
                    }
                }
            }
            pooledData[i][j] = max;
        }
    }
    return pooledData;
}

function averagePooling(inputData, poolSize) {
    const size = inputData.length;
    const pooledSize = Math.floor(size / poolSize);

    const pooledData = new Array(pooledSize);
    for (let i = 0; i < pooledSize; i++) {
        pooledData[i] = new Array(pooledSize);
        for (let j = 0; j < pooledSize; j++) {
            let sum = 0;
            for (let m = 0; m < poolSize; m++) {
                for (let n = 0; n < poolSize; n++) {
                    sum += inputData[i * poolSize + m][j * poolSize + n];
                }
            }
            const average = sum / (poolSize * poolSize);
            pooledData[i][j] = average;
        }
    }
    return pooledData;
}

function displayPooledData(data) {
    const outputDiv = document.getElementById('output');
    outputDiv.innerHTML = '<strong>Feature map after Pooling operation:</strong><br>';
    for (let i = 0; i < data.length; i++) {
        outputDiv.innerHTML += data[i].join(' ') + '<br>';
    }
}

document.addEventListener('DOMContentLoaded', createInputTable);