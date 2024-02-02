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
    <script src="\files\html\convolution.js" defer></script>
</body>
</html>

