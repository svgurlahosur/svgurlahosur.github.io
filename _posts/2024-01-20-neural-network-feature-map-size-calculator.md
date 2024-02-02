---
title: 'Convolution and Pooling operation feature map shape calculator.'
date: 2024-01-20
permalink: /posts/2024/02/neural-network-feature-map-size-calculator/
excerpt_separator: <!--more-->
toc: false
tags:
  - filter/kernel
  - convolution
  - stride
  - padding
  - feature maps
  - pooling
  - CNN
---

This post allows users calculate/track the feature map size while stacking the Convolution and Pooling operations and select the appropriate number of neurons in the first Fully Connected/ Dense/Linear layer while building Convolutional Neural Networks.


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
            width: 250px;
        }
        .input {
            width: 50px;
        }
        .large-button {
            font-size: 17px;
            padding: 6px 15px;
            white-space: nowrap;
            width: 590px;
            text-align: center;
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
            width: 250 px;
        }
        table, th, td {
            border: 1px solid black;
            padding: 5px;
        }
        .row.left-align-button {
            display: flex;
            justify-content: flex-start;
            gap: 280px;
        }
        .input button {
            margin-right: 15px; /* Adjust the space around the buttons */
        }
        table {
            width: 350px; /* Adjust the width as needed */
        }
        table td {
            padding: 5px;
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
            width: 100px; /* Adjust the width as needed */
            margin-right: 20px; /* Creating a gap between input fields */
        }
    </style>
    <script>
        window.onload = function() {
            selectOperation(); // To show the appropriate options based on the default selection
        }
    </script>
</head>
<body class="left-align">
    <div class="row">
        <div class="label">
            <strong>Input Data height/width:</strong>
        </div>
        <div class="input">
            <input type="number" id="inputSize" value="4" min="1" max="10">
        </div>
        <div class="label">
            <strong>Number of input channels:</strong>
        </div>
        <div class="input">
            <input type="number" id="numChannels" value="1" min="1" max="10">
        </div>
    </div>
    <div id="operationSelector" class="row">
        <div class="label">
            <strong>Select the Operation/layer:</strong>
        </div>
        <div class="input">
            <select id="operationType" onchange="selectOperation()">
                <option value="convolution" selected>Convolution</option>
                <option value="pooling">Pooling</option>
            </select>
        </div>
    </div>
    <div id="convolutionOptions">
        <div class="row">
            <table>
                <tr>
                    <td><strong>Filter Size:</strong></td>
                    <td><input type="number" id="filterSizeValue" value="3" min="1" max="10"></td>
                </tr>
                <tr>
                    <td><strong>Stride:</strong></td>
                    <td><input type="number" id="strideValue" value="1" min="1" max="10"></td>
                </tr>
                <tr>
                    <td><strong>Padding:</strong></td>
                    <td><input type="number" id="paddingValue" value="0" min="0" max="10"></td>
                </tr>
                <tr>
                    <td><strong>Number of Filters:</strong></td>
                    <td><input type="number" id="numFiltersValue" value="1" min="1" max="10"></td>
                </tr>
            </table>
        </div>
    </div>
    <div id="poolingOptions" style="display: none;">
        <div class="row">
            <table>
                <tr>
                    <td><strong>Pooling Filter Size:</strong></td>
                    <td><input type="number" id="poolingFilterSize" value="2" min="1" max="10"></td>
                </tr>
            </table>
        </div>
    </div>
    <div class="row left-align-button">
        <div class="input">
            <button class="large-button" onclick="calculateFeatureMap()"><strong>Calculate the feature map size and load it as current feature map size</strong></button>
        </div>
    </div>
    <div id="output"></div>
    <script src="\files\html\feature_map.js" defer></script>
</body>
</html>