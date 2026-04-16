let currentFilePath = '';

// Get filepath when page loads
document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    currentFilePath = urlParams.get('filepath');
});

// Function to show loading animation
function showLoading() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loadingAnimation').style.display = 'block';
    document.getElementById('visualizationsGrid').innerHTML = ''; // Clear existing visualizations
}

// Function to hide loading animation
function hideLoading() {
    document.getElementById('loadingAnimation').style.display = 'none';
}

// Function to show error message
function showError(message) {
    hideLoading();
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    document.getElementById('visualizationsGrid').appendChild(errorDiv);
}

// Chart generation function
document.getElementById('generateChart').addEventListener('click', async function() {
    // Get selected columns
    const selectedColumns = Array.from(document.querySelectorAll('input[name="column"]:checked'))
        .map(checkbox => checkbox.value);
    
    // Get selected chart types
    const selectedChartTypes = Array.from(document.querySelectorAll('input[name="chartType"]:checked'))
        .map(checkbox => checkbox.value);
    
    // Get sample percentage
    const samplePercentage = parseInt(document.getElementById('dataSamplePercentage').value);
    
    // Validate selections
    if (selectedColumns.length === 0) {
        alert('Please select at least one column');
        return;
    }
    
    if (selectedChartTypes.length === 0) {
        alert('Please select at least one chart type');
        return;
    }

    if (!currentFilePath) {
        alert('No data file selected');
        return;
    }

    // Show loading animation
    showLoading();
    
    try {
        // Create visualization grid container
        const gridContainer = document.getElementById('visualizationsGrid');
        gridContainer.innerHTML = ''; // Clear existing content
        
        // Generate charts for each selected type
        for (const chartType of selectedChartTypes) {
            try {
                const response = await fetch('/generate_visualization', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        filepath: currentFilePath,
                        columns: selectedColumns,
                        type: chartType,
                        sample_percentage: samplePercentage
                    })
                });

                const data = await response.json();
                
                if (data.success) {
                    // Create visualization container
                    const vizContainer = document.createElement('div');
                    vizContainer.className = 'visualization-item';
                    vizContainer.id = `viz-${chartType}`;
                    gridContainer.appendChild(vizContainer);

                    // The visualization data is already a JSON object, no need to parse
                    const vizData = data.visualization;
                    
                    // Create layout with responsive dimensions
                    const layout = {
                        ...vizData.layout,
                        width: 800,
                        height: 600,
                        autosize: true,
                        margin: { l: 50, r: 30, t: 50, b: 50 },
                        font: { size: 12 },
                        showlegend: true
                    };
                    
                    // Render the visualization using Plotly with the layout
                    await Plotly.newPlot(vizContainer, vizData.data, layout, {
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['sendDataToCloud'],
                        toImageButtonOptions: {
                            format: 'png',
                            filename: 'visualization',
                            height: 600,
                            width: 800,
                            scale: 2
                        }
                    });
                    
                    // Enable export buttons
                    document.getElementById('exportPNG').disabled = false;
                    document.getElementById('exportPDF').disabled = false;
                    document.getElementById('addToDashboard').disabled = false;
                } else {
                    showError(`Error generating ${chartType} chart: ${data.error}`);
                }
            } catch (error) {
                console.error(`Error generating ${chartType} chart:`, error);
                showError(`Failed to generate ${chartType} chart`);
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to generate visualizations');
    } finally {
        hideLoading();
    }
}); 