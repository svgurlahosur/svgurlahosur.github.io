---
title: 'Convolution operation feature map calculator.'
date: 2023-09-23
permalink: /posts/2023/02/convolution-feature-map-calculator/
excerpt_separator: <!--more-->
toc: false
tags:
  - filter/kernel
  - convolution
  - stride
  - padding
  - feature maps
  - input
  - CNN
---

This post allows users to experiment with 2D convolution by defining input data shape, values, filter/kernel shape, values, and other parameters like stride and padding. It will calculate the feature map size and values for the given input and kernel values.

<!--more-->

<html>
<head>
<style>
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
</style>
</head>
<body class="left-align">
    <div class="row">
        <div class="label">
            <strong>Input Data Height:</strong>
        </div>
        <div class="input">
            <input type="number" id="inputHeight" value="1" min="1" max="10" oninput="updateOutputSize()">
        </div>
        <div class="label">
            <strong>&nbsp;&nbsp;Input Data Width:</strong>
        </div>
        <div class="input">
            <input type="number" id="inputWidth" value="1" min="1" max="10" oninput="updateOutputSize()">
        </div>
    </div>
    <div class="row">
        <div class="label">
            <strong>Filter Data Height:</strong>
        </div>
        <div class="input">
            <input type="number" id="filterHeight" value="1" min="1" max="10" oninput="updateOutputSize()">
        </div>
        <div class="label">
            <strong>&nbsp;&nbsp;Filter Data Width:</strong>
        </div>
        <div class="input">
            <input type="number" id="filterWidth" value="1" min="1" max="10" oninput="updateOutputSize()">
        </div>
    </div>
    <div class="row">
        <div class="label">
            <strong>Padding:</strong>
        </div>
        <div class="input">
            <input type="number" id="padding" value="0" min="0" max="10" oninput="updateOutputSize()">
        </div>
        <div class="label">
            <strong>&nbsp;&nbsp;Stride:</strong>
        </div>
        <div class="input">
            <input type="number" id="stride" value="1" min="1" max="10" oninput="updateOutputSize()">
        </div>
    </div>
    <div id="outputSize"></div>
    <div class="row left-align-button">
        <div class="input">
            <button class="large-button" onclick="createInputFields()"><strong>Enter values for input and filter</strong></button>
        </div>
    </div>
    <div id="inputDataFields"></div>
    <div id="filterDataFields"></div>
    <div class="row left-align-button">
        <div class="input">
            <button class="large-button" onclick="calculateConvolution()"><strong>Calculate feature map values</strong></button>
        </div>
    </div>
    <div id="output"></div>
    <script>
        function updateOutputSize() {
            const inputHeight = parseInt(document.getElementById('inputHeight').value);
            const inputWidth = parseInt(document.getElementById('inputWidth').value);
            const filterHeight = parseInt(document.getElementById('filterHeight').value);
            const filterWidth = parseInt(document.getElementById('filterWidth').value);
            const padding = parseInt(document.getElementById('padding').value);
            const stride = parseInt(document.getElementById('stride').value);
            const outputHeight = Math.floor((inputHeight - filterHeight + 2 * padding) / stride) + 1;
            const outputWidth = Math.floor((inputWidth - filterWidth + 2 * padding) / stride) + 1;
            const outputSizeDiv = document.getElementById('outputSize');
            outputSizeDiv.innerHTML = '<strong>Feature Map Size:</strong><br>';
            outputSizeDiv.innerHTML += `Height: ${outputHeight}<br>`;
            outputSizeDiv.innerHTML += `Width: ${outputWidth}`;
        }
        function createInputFields() {
            const inputHeight = parseInt(document.getElementById('inputHeight').value);
            const inputWidth = parseInt(document.getElementById('inputWidth').value);
            const filterHeight = parseInt(document.getElementById('filterHeight').value);
            const filterWidth = parseInt(document.getElementById('filterWidth').value);
            const inputDataFields = document.getElementById('inputDataFields');
            const filterDataFields = document.getElementById('filterDataFields');
            inputDataFields.innerHTML = '<strong>Input Data:</strong><br>';
            filterDataFields.innerHTML = '<strong>Filter Data:</strong><br>';
            const createTable = (container, rows, cols, idPrefix) => {
                const table = document.createElement('table');
                for (let i = 0; i < rows; i++) {
                    const row = document.createElement('tr');
                    for (let j = 0; j < cols; j++) {
                        const cell = document.createElement('td');
                        const input = document.createElement('input');
                        input.type = 'number';
                        input.id = `${idPrefix}_${i}_${j}`;
                        input.value = "1"; // Set default value to 1
                        input.min = "-5"; // Set minimum value to 1
                        input.max = "10"; // Set maximum value to 10
                        cell.appendChild(input);
                        row.appendChild(cell);
                    }
                    table.appendChild(row);
                }
                container.appendChild(table);
            }
            createTable(inputDataFields, inputHeight, inputWidth, 'input');
            createTable(filterDataFields, filterHeight, filterWidth, 'filter');
        }
        function calculateConvolution() {
            const inputHeight = parseInt(document.getElementById('inputHeight').value);
            const inputWidth = parseInt(document.getElementById('inputWidth').value);
            const filterHeight = parseInt(document.getElementById('filterHeight').value);
            const filterWidth = parseInt(document.getElementById('filterWidth').value);
            const padding = parseInt(document.getElementById('padding').value);
            const stride = parseInt(document.getElementById('stride').value);
            const inputData = new Array(inputHeight);
            for (let i = 0; i < inputHeight; i++) {
                inputData[i] = new Array(inputWidth);
                for (let j = 0; j < inputWidth; j++) {
                    inputData[i][j] = parseFloat(document.getElementById(`input_${i}_${j}`).value);
                }
            }
            const filterData = new Array(filterHeight);
            for (let i = 0; i < filterHeight; i++) {
                filterData[i] = new Array(filterWidth);
                for (let j = 0; j < filterWidth; j++) {
                    filterData[i][j] = parseFloat(document.getElementById(`filter_${i}_${j}`).value);
                }
            }
            const output = convolution2D(inputData, filterData, padding, stride);
            const outputDiv = document.getElementById('output');
            outputDiv.innerHTML = '<strong>Feature Map values:</strong><br>';
            for (let i = 0; i < output.length; i++) {
                outputDiv.innerHTML += output[i].join(' ') + '<br>';
            }
        }
        function convolution2D(inputData, filterData, padding, stride) {
            const inputHeight = inputData.length;
            const inputWidth = inputData[0].length;
            const filterHeight = filterData.length;
            const filterWidth = filterData[0].length;
            const outputHeight = Math.floor((inputHeight - filterHeight + 2 * padding) / stride) + 1;
            const outputWidth = Math.floor((inputWidth - filterWidth + 2 * padding) / stride) + 1;
            const output = new Array(outputHeight);
            for (let i = 0; i < outputHeight; i++) {
                output[i] = new Array(outputWidth).fill(0);
            }
            const paddedInputData = new Array(inputHeight + 2 * padding);
            for (let i = 0; i < inputHeight + 2 * padding; i++) {
                paddedInputData[i] = new Array(inputWidth + 2 * padding).fill(0);
            }
            for (let i = 0; i < inputHeight; i++) {
                for (let j = 0; j < inputWidth; j++) {
                    paddedInputData[i + padding][j + padding] = inputData[i][j];
                }
            }
            for (let i = 0; i < outputHeight; i++) {
                for (let j = 0; j < outputWidth; j++) {
                    let sum = 0;
                    for (let m = 0; m < filterHeight; m++) {
                        for (let n = 0; n < filterWidth; n++) {
                            sum += paddedInputData[i * stride + m][j * stride + n] * filterData[m][n];
                        }
                    }
                    output[i][j] = sum;
                }
            }
            return output;
        }
    </script>
</body>
</html>

