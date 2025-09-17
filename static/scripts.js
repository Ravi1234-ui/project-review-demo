document.addEventListener('DOMContentLoaded', () => {
    const inputField = document.querySelector('input[name="stock"]');
    const tickerSelect = document.getElementById('ticker_select');
    const submitButton = document.querySelector('input[type="submit"]');
    const loading = document.getElementById('loading');
    const themeToggle = document.getElementById('themeToggle');

    // Uppercase automatically
    inputField.addEventListener('input', () => {
        inputField.value = inputField.value.toUpperCase();
        if (tickerSelect.value !== '') {
            tickerSelect.value = '';
        }
    });

    // Sync dropdown → input
    tickerSelect.addEventListener('change', () => {
        if (tickerSelect.value !== '') {
            inputField.value = tickerSelect.value;
        }
    });

    // Show spinner + scroll to charts after submit
    submitButton.addEventListener('click', () => {
        if (inputField.value.trim() === '') {
            alert('⚠️ Please enter a stock symbol or choose from the list.');
        } else {
            loading.style.display = 'block';
            setTimeout(() => {
                const charts = document.getElementById('chartsSection');
                if (charts) {
                    charts.scrollIntoView({ behavior: 'smooth' });
                }
                loading.style.display = 'none';
            }, 3000); // fake delay for UX
        }
    });

    // Dark mode toggle
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        themeToggle.textContent = document.body.classList.contains('dark-mode')
            ? '☀️ Light Mode'
            : '🌙 Dark Mode';
    });
});
