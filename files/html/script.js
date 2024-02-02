function calculateArea() {
    // Get the values from the input fields
    var length = parseFloat(document.getElementById('length').value);
    var width = parseFloat(document.getElementById('width').value);

    // Check if the values are valid numbers
    if (isNaN(length) || isNaN(width)) {
        document.getElementById('result').textContent = 'Please enter valid numbers for length and width.';
        return;
    }

    // Calculate the area
    var area = length * width;

    // Display the result
    document.getElementById('result').textContent = 'The area of the rectangle is: ' + area.toFixed(2);
}
