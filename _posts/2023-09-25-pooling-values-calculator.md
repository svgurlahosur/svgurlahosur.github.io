---
title: 'Pooling operation feature map calculator.'
date: 2023-09-25
permalink: /posts/2023/03/pooling-values-calculator/
excerpt_separator: <!--more-->
toc: false
tags:
  - pooling
  - max pooling
  - average pooling
  - feature maps
  - CNN
---

This post allows users to experiment with 2D pooling by defining input data shape, values, and pooling filter/kernel shape. It will calculate the feature map size and values for the given input based on max or average pooling.


<!--more-->

<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }

        .row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }

        .label {
            width: 200px;
        }

        .input {
            width: 50px;
        }

        .large-button {
            font-size: 16px;
            padding: 6px 15px;
            white-space: nowrap;
            width: 280px;
            text-align: left;
        }

        .left-align {
            text-align: left;
        }

        .left-align-button {
            text-align: left;
        }

        table {
            border-collapse: collapse;
            margin-top: 10px;
        }

        table, th, td {
            border: 1px solid black;
            padding: 5px;
        }

        .row.left-align-button {
            display: flex;
            justify-content: flex-start;
            gap: 250px;
        }

        .input button {
            margin-right: 15px; /* Adjust the space around the buttons */
    </style>

</head>


<body class="left-align">
    <div class="row">
        <div class="label">
            <strong>Input Data Height/Width:</strong>
        </div>
        <div class="input">
            <input type="number" id="inputSize" value="4" min="1" max="10" onchange="createInputTable()">
        </div>
    </div>
    <div id="inputTable"></div>
    <div id="poolingInputs" style="display: none;">
        <div class="row">
            <div class="label">
                <strong>Pooling Filter Height/Width:</strong>
            </div>
            <div class="input">
                <input type="number" id="poolSize" value="2" min="1" max="5">
            </div>
        </div>
        <div class="row left-align-button">
            <div class="input">
                <button class="large-button" onclick="performMaxPooling()"><strong>Perform Max Pooling</strong></button>
            </div>
            <div class="input">
                <button class="large-button" onclick="performAveragePooling()"><strong>Perform Average Pooling</strong></button>
            </div>
        </div>
    </div>
    <div id="output"></div>
    <script>
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
    </script>
</body>
</html>
