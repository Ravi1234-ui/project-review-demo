document.addEventListener('DOMContentLoaded', () => {
    const inputField = document.querySelector('input[name="stock"]');
    const submitButton = document.querySelector('input[type="submit"]');

    // Automatically uppercase the stock symbol on input
    inputField.addEventListener('input', () => {
        inputField.value = inputField.value.toUpperCase();
    });

    // Simple alert on form submission
    submitButton.addEventListener('click', () => {
        if (inputField.value.trim() === '') {
            alert('Please enter a stock symbol.');
        } else {
            console.log('Form submitted for stock:', inputField.value);
        }
    });
});
