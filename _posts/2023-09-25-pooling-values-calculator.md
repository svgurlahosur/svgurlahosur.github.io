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
        }
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
    <script src="\files\html\pooling.js" defer></script>
</body>
</html>
