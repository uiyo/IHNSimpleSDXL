
var dropdown = gradioApp().getElementById('refiner_model');
dropdown.value = 'artbookv2.stafetensors';
dropdown.addEventListener('change', function() {
    var selectedOption = this.value;
    fetch('/api/update-dropdown', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            choice: selectedOption
        })
    })
    .then(response => response.json())
    .then(data => {
        // 根据返回的数据更新Dropdown的选项
        // ...
    })
    .catch(error => {
        console.error('Error:', error);
    });
});