import { pipeline, AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure environment
env.allowLocalModels = false;
env.useBrowserCache = true;

// DOM elements
const textInput = document.getElementById("textinput");
const submitButton = document.getElementById("submit");
const outputElement = document.getElementById("output");
const tokensDisplay = document.getElementById("tokensDisplay");
const loadingContainer = document.getElementById("loading-container");
const appContainer = document.getElementById("app-container");
const pcaContainer = document.getElementById("pca-container");
const pcaComponentsElement = document.getElementById("pca-components");
const pcaGridsContainer = document.getElementById("pca-grids-container");
const currentWeightsElement = document.getElementById("current-weights");
const processingIndicator = document.getElementById("processing-indicator");
const grid12Canvas = document.getElementById("grid-1-2");
const grid34Canvas = document.getElementById("grid-3-4");
const grid56Canvas = document.getElementById("grid-5-6");

// Initialize model and tokenizer
let model;
let tokenizer;

// Load model and tokenizer
(async function loadModel() {
    try {
        model = await pipeline('feature-extraction', 'Xenova/bert-base-uncased', { revision: 'default' });
        tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased', { revision: 'default' });
        
        loadingContainer.classList.add('hidden');
        appContainer.style.opacity = '1';
        outputElement.textContent = "Enter text and click Submit.";
        submitButton.disabled = false;
        
        // Initialize the tokens display with a placeholder to maintain layout
        tokensDisplay.innerHTML = '<span style="color: #999; font-style: italic;">Tokens will appear here after processing...</span>';
    } catch (error) {
        console.error("Error loading model or tokenizer:", error);
        loadingContainer.textContent = "Error loading model: " + error.message;
    }
})();

// Helper functions for PCA
function dotProduct(v1, v2) {
    return v1.reduce((sum, val, idx) => sum + val * v2[idx], 0);
}

function normalize(vector) {
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return magnitude > 0 ? vector.map(val => val / magnitude) : vector.map(() => 1 / Math.sqrt(vector.length));
}

// Calculate multiple PCA components
function calculatePCA(vectors, numComponents = 5) {
    if (!vectors || vectors.length <= 1) {
        return { componentVectors: [], projections: [], explainedVariance: [], mean: [] };
    }
    
    const numVectors = vectors.length;
    const dimension = vectors[0].length;
    numComponents = Math.min(numComponents, numVectors - 1, dimension);
    
    // Calculate mean vector
    const mean = vectors[0].map((_, j) => vectors.reduce((sum, v) => sum + v[j], 0) / numVectors);
    
    // Center the data
    const centered = vectors.map(v => v.map((x, j) => x - mean[j]));
    
    // Calculate total variance
    const totalVariance = centered.reduce((sum, v) => sum + v.reduce((s, x) => s + x * x, 0), 0) || 1e-10;
    
    // Power iteration with deflation
    const componentVectors = [];
    const projections = [];
    const explainedVariance = [];
    let residuals = JSON.parse(JSON.stringify(centered)); // Deep copy
    
    for (let comp = 0; comp < numComponents; comp++) {
        // Initialize random vector
        let eigenVector = Array(dimension).fill().map(() => Math.random() * 2 - 1);
        eigenVector = normalize(eigenVector);
        
        // Power iteration
        for (let iter = 0; iter < 20; iter++) {
            let newVector = residuals[0].map(() => 0);
            for (let i = 0; i < numVectors; i++) {
                const dp = dotProduct(residuals[i], eigenVector);
                for (let j = 0; j < dimension; j++) {
                    newVector[j] += residuals[i][j] * dp;
                }
            }
            eigenVector = normalize(newVector);
        }
        
        // Compute projections
        const proj = centered.map(v => dotProduct(v, eigenVector));
        
        // Compute component variance
        const componentVariance = proj.reduce((sum, p) => sum + p * p, 0);
        
        // Normalize projections to [-1, 1]
        const maxAbs = Math.max(...proj.map(Math.abs), 1e-10);
        const normalizedProj = proj.map(p => Math.max(Math.min(p / maxAbs, 1), -1));
        
        // Store results
        componentVectors.push(eigenVector);
        projections.push(normalizedProj);
        explainedVariance.push(componentVariance / totalVariance);
        
        // Deflate
        for (let i = 0; i < numVectors; i++) {
            const dp = dotProduct(residuals[i], eigenVector);
            for (let j = 0; j < dimension; j++) {
                residuals[i][j] -= dp * eigenVector[j];
            }
        }
    }
    
    return { componentVectors, projections, explainedVariance, mean };
}

// Global variables for component switching and interactive grid
let currentTokens = [];
let currentPcaResult = null;
let selectedComponent = 0;
let compositeWeights = [1, 0, 0, 0, 0, 0]; // Default weights for the 6 components
let tokenEmbeddings = []; // Store original token embeddings
let savedVectors = JSON.parse(localStorage.getItem('savedVectors')) || [];
let selectedMode = 'pca'; // Possible values: 'pca', 'grids', 'saved', 'custom'
let selectedSavedVectorIndex = -1;
let selectedTokenIndices = []; // Track selected token indices
let currentTempVector = null; // Temporary vector for selection without saving

// Grid variables
const gridPoints = [
    { x: 100, y: 100, hasBeenMoved: false }, // Grid 1-2 point (center)
    { x: 100, y: 100, hasBeenMoved: false }, // Grid 3-4 point (center)
    { x: 100, y: 100, hasBeenMoved: false }  // Grid 5-6 point (center)
];

// Convert projection value to color (red to blue gradient)
function valueToColor(value) {
    const normalizedValue = (value + 1) / 2; // Map [-1, 1] to [0, 1]
    const r = Math.floor(255 * (1 - normalizedValue));
    const b = Math.floor(255 * normalizedValue);
    return `rgb(${r}, 0, ${b})`; // Red to blue gradient through purple
}

// Calculate Euclidean distance between two vectors
function euclideanDistance(vecA, vecB) {
    let sumSquaredDiffs = 0;
    
    for (let i = 0; i < vecA.length; i++) {
        const diff = vecA[i] - vecB[i];
        sumSquaredDiffs += diff * diff;
    }
    
    return Math.sqrt(sumSquaredDiffs);
}

// Create composite vector from weights and component vectors
function createCompositeVector(componentVectors, weights) {
    if (!componentVectors.length) return [];
    
    const dimension = componentVectors[0].length;
    const compositeVector = new Array(dimension).fill(0);
    
    // Combine components using weights
    for (let i = 0; i < Math.min(componentVectors.length, weights.length); i++) {
        for (let j = 0; j < dimension; j++) {
            compositeVector[j] += componentVectors[i][j] * weights[i];
        }
    }
    
    // Normalize the composite vector
    const magnitude = Math.sqrt(compositeVector.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
        for (let i = 0; i < dimension; i++) {
            compositeVector[i] /= magnitude;
        }
    }
    
    return compositeVector;
}

// Draw a grid with a draggable point
function drawGrid(canvas, pointIndex) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const point = gridPoints[pointIndex];
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw grid lines
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    
    // Vertical and horizontal lines
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    
    // Draw point
    ctx.fillStyle = 'blue';
    ctx.beginPath();
    ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw vector line from center to point
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(width / 2, height / 2);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
}

// Draw all grids
function drawAllGrids() {
    drawGrid(grid12Canvas, 0);
    drawGrid(grid34Canvas, 1);
    drawGrid(grid56Canvas, 2);
}

// Set up grid event listeners for dragging
function setupGridListeners() {
    setupGridDragging(grid12Canvas, 0, 0, 1);
    setupGridDragging(grid34Canvas, 1, 2, 3);
    setupGridDragging(grid56Canvas, 2, 4, 5);
    
    // Make the entire grid container clickable for selecting custom vector
    pcaGridsContainer.addEventListener('click', (e) => {
        // Only handle clicks outside the canvas elements
        if (e.target.tagName.toLowerCase() === 'canvas') {
            return; // Let the canvas handle its own clicks
        }
        
        selectedMode = 'grids';
        selectedComponent = -1;
        updateSelectionUI();
        
        updateWeightsFromGrids();
        
        if (currentPcaResult && currentPcaResult.componentVectors.length >= 6) {
            colorTokens();
        }
    });
}

// Create drag functionality for a grid
function setupGridDragging(canvas, pointIndex, xComponentIndex, yComponentIndex) {
    let isDragging = false;
    
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Check if click is close to the point
        const point = gridPoints[pointIndex];
        const distance = Math.sqrt((x - point.x) ** 2 + (y - point.y) ** 2);
        
        if (distance < 10) {
            isDragging = true;
        } else {
            // Set point to clicked position
            point.x = x;
            point.y = y;
            point.hasBeenMoved = true;
            updateWeightsFromGrids();
            drawAllGrids();
        }
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
        const y = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));
        
        gridPoints[pointIndex].x = x;
        gridPoints[pointIndex].y = y;
        gridPoints[pointIndex].hasBeenMoved = true;
        
        updateWeightsFromGrids();
        drawAllGrids();
    });
    
    canvas.addEventListener('mouseup', () => {
        isDragging = false;
    });
    
    canvas.addEventListener('mouseleave', () => {
        isDragging = false;
    });
}

// Update component weights from grid positions
function updateWeightsFromGrids() {
    // For each grid, compute the vector from center to point
    for (let i = 0; i < gridPoints.length; i++) {
        const point = gridPoints[i];
        const xComponent = i * 2;
        const yComponent = i * 2 + 1;
        
        // Convert grid coordinates to -1 to 1 range
        const xWeight = (point.x - 100) / 100; // Center is at 100,100
        const yWeight = (100 - point.y) / 100; // Y is inverted
        
        compositeWeights[xComponent] = xWeight;
        compositeWeights[yComponent] = yWeight;
    }
    
    // Update display
    currentWeightsElement.textContent = compositeWeights.map(w => w.toFixed(2)).join(', ');
    
    // Update selection state
    selectedMode = 'grids';
    selectedComponent = -1;
    updateSelectionUI();
    
    // Update token coloring if we have embeddings
    if (currentPcaResult && currentPcaResult.componentVectors.length >= 6) {
        colorTokens();
    }
}

// Render the saved vectors list
function renderSavedVectors() {
    const listElement = document.getElementById('saved-vectors-list');
    listElement.innerHTML = '';
    
    // Show or hide the container based on whether we have vectors
    const savedVectorsContainer = document.getElementById('saved-vectors-container');
    if (savedVectors.length > 0) {
        savedVectorsContainer.classList.remove('hidden');
    } else {
        savedVectorsContainer.classList.add('hidden');
        return;
    }
    
    savedVectors.forEach((vec, index) => {
        const div = document.createElement('div');
        div.className = `saved-vector ${selectedMode === 'saved' && selectedSavedVectorIndex === index ? 'selected' : ''}`;
        div.dataset.index = index;
        div.innerHTML = `
            <span class="vector-name">${vec.name}</span>
            <div class="vector-actions">
                <button class="rename-vector">Rename</button>
                <button class="delete-vector">Delete</button>
            </div>
        `;
        div.addEventListener('click', (e) => {
            // Don't trigger selection when clicking buttons
            if (e.target.tagName.toLowerCase() === 'button') {
                return;
            }
            selectedMode = 'saved';
            selectedSavedVectorIndex = index;
            updateSelectionUI();
            colorTokens();
        });
        listElement.appendChild(div);
    });
    
    document.querySelectorAll('.rename-vector').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.target.closest('.saved-vector').dataset.index);
            showPrompt('Enter new name:', savedVectors[index].name, (newName) => {
                if (newName) {
                    savedVectors[index].name = newName;
                    localStorage.setItem('savedVectors', JSON.stringify(savedVectors));
                    renderSavedVectors();
                }
            });
        });
    });
    
    document.querySelectorAll('.delete-vector').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.target.closest('.saved-vector').dataset.index);
            savedVectors.splice(index, 1);
            localStorage.setItem('savedVectors', JSON.stringify(savedVectors));
            renderSavedVectors();
            if (selectedMode === 'saved' && selectedSavedVectorIndex === index) {
                selectedMode = 'pca';
                selectedComponent = 0;
                selectedSavedVectorIndex = -1;
                updateSelectionUI();
                colorTokens();
            }
        });
    });
}

// Update the UI based on the selected mode
function updateSelectionUI() {
    // Remove selected class from all selectable elements
    document.querySelectorAll('.pca-component').forEach(el => el.classList.remove('selected'));
    pcaGridsContainer.classList.remove('selected');
    document.querySelectorAll('.saved-vector').forEach(el => el.classList.remove('selected'));
    
    // Add selected class based on current mode
    if (selectedMode === 'pca') {
        const componentEl = document.querySelector(`.pca-component[data-component-index="${selectedComponent}"]`);
        if (componentEl) componentEl.classList.add('selected');
    } else if (selectedMode === 'grids') {
        pcaGridsContainer.classList.add('selected');
    } else if (selectedMode === 'saved' && selectedSavedVectorIndex >= 0) {
        const savedEl = document.querySelector(`.saved-vector[data-index="${selectedSavedVectorIndex}"]`);
        if (savedEl) savedEl.classList.add('selected');
    }
}

// Unified function for token coloring
function colorTokens() {
    if (!currentTokens.length || !tokenEmbeddings.length || !currentPcaResult) return;
    
    let similarities;
    
    if (selectedMode === 'pca') {
        // Use precomputed PCA projections
        similarities = currentPcaResult.projections[selectedComponent];
    } else {
        // Calculate similarities using dot product with vector
        let vector;
        
        if (selectedMode === 'grids') {
            // Create vector from component weights
            vector = createCompositeVector(
                currentPcaResult.componentVectors.slice(0, 6),
                compositeWeights
            );
        } else if (selectedMode === 'saved' && selectedSavedVectorIndex >= 0) {
            // Use saved vector
            vector = savedVectors[selectedSavedVectorIndex].vector;
        } else if (selectedMode === 'custom' && currentTempVector) {
            // Use temporary vector from selection
            vector = currentTempVector;
        } else {
            return; // No valid mode selected
        }
        
        // Get mean from PCA result for consistent centering
        const mean = currentPcaResult.mean;
        
        // Calculate similarity for each token
        similarities = tokenEmbeddings.map(embedding => {
            // Center the embedding using the mean
            const centered = embedding.map((val, idx) => val - mean[idx]);
            // Calculate dot product with the vector
            return dotProduct(centered, vector);
        });
    }
    
    // Normalize similarities to [-1, 1] range
    const maxAbs = Math.max(...similarities.map(Math.abs), 1e-10);
    const normalizedSimilarities = similarities.map(s => Math.max(Math.min(s / maxAbs, 1), -1));
    
    // Generate HTML for tokens
    let tokensHTML = "";
    const punctuation = [',', '.', ':', ';', '!', '?', ')', ']', '}', "'", '"'];
    
    currentTokens.forEach((token, i) => {
        const color = valueToColor(normalizedSimilarities[i] || 0);
        const isSelected = selectedTokenIndices.includes(i);
        let html = `<span class="token ${isSelected ? 'selected' : ''}" data-index="${i}" style="color: ${color};">`;
        
        // Handle BERT subword tokens that start with ##
        if (token.startsWith('##')) {
            html += `${token.substring(2)}</span>`;
        } 
        // Don't add space before punctuation
        else if (punctuation.includes(token) || punctuation.includes(token.trim())) {
            html += `${token}</span>`;
        }
        // Handle tokens that are just spaces
        else if (token === ' ' || token === '  ') {
            html += `${token}</span>`;
        }
        // Add space before normal tokens (but not at the beginning)
        else if (i > 0) {
            // Only add space if the token doesn't already start with one
            const needsSpace = !token.startsWith(' ');
            html += `${needsSpace ? ' ' : ''}${token}</span>`;
        } 
        // First token
        else {
            html += `${token}</span>`;
        }
        
        tokensHTML += html;
    });
    
    tokensDisplay.innerHTML = tokensHTML;
    
    // Add click handlers to tokens for selection
    document.querySelectorAll('.token').forEach(tokenEl => {
        tokenEl.addEventListener('click', (e) => {
            const index = parseInt(e.target.dataset.index);
            e.target.classList.toggle('selected');
            
            // Update selectedTokenIndices
            if (selectedTokenIndices.includes(index)) {
                selectedTokenIndices = selectedTokenIndices.filter(i => i !== index);
            } else {
                selectedTokenIndices.push(index);
            }
            
            // Show or hide Find Vector button based on selection
            document.getElementById('find-vector').classList.toggle('hidden', selectedTokenIndices.length === 0);
        });
    });
}

// Render PCA component selectors
function renderPcaSelectors(pcaResult) {
    pcaComponentsElement.innerHTML = '';
    pcaResult.explainedVariance.forEach((variance, index) => {
        const percentVariance = (variance * 100).toFixed(1);
        const componentDiv = document.createElement('div');
        componentDiv.className = `pca-component ${index === selectedComponent ? 'selected' : ''}`;
        componentDiv.dataset.componentIndex = index;
        componentDiv.innerHTML = `
            <span class="component-label">Component ${index + 1}</span>
            <span class="variance-explained">${percentVariance}%</span>
        `;
        componentDiv.addEventListener('click', () => {
            selectedMode = 'pca';
            selectedComponent = index;
            updateSelectionUI();
            colorTokens();
        });
        pcaComponentsElement.appendChild(componentDiv);
    });
    pcaContainer.classList.remove('hidden');
    
    // Initialize the grid visualization if we have enough components
    if (pcaResult.componentVectors.length >= 6) {
        pcaGridsContainer.classList.remove('hidden');
        setupGridListeners();
        drawAllGrids();
    } else {
        pcaGridsContainer.classList.add('hidden');
    }
}

// Save current vector event listener
document.getElementById('save-vector').addEventListener('click', () => {
    if (!currentPcaResult) {
        showAlert('Please process some text first.');
        return;
    }
    
    showPrompt('Enter name for the vector:', '', (name) => {
        if (!name) return;
        
        let vector;
        if (selectedMode === 'pca') {
            vector = currentPcaResult.componentVectors[selectedComponent];
        } else if (selectedMode === 'grids') {
            vector = createCompositeVector(
                currentPcaResult.componentVectors.slice(0, 6),
                compositeWeights
            );
        } else if (selectedMode === 'saved') {
            vector = savedVectors[selectedSavedVectorIndex].vector;
        } else if (selectedMode === 'custom' && currentTempVector) {
            vector = currentTempVector;
        } else {
            showAlert('No vector selected to save.');
            return;
        }
        
        // Add the new vector to savedVectors
        savedVectors.push({ name, vector });
        localStorage.setItem('savedVectors', JSON.stringify(savedVectors));
        
        // Update the UI
        renderSavedVectors();
        
        // Switch to the newly saved vector
        selectedMode = 'saved';
        selectedSavedVectorIndex = savedVectors.length - 1;
        updateSelectionUI();
    });
});

// Find vector from selection event listener
document.getElementById('find-vector').addEventListener('click', () => {
    if (selectedTokenIndices.length === 0) {
        showAlert('Please select some tokens first.');
        return;
    }
    
    const unselectedIndices = currentTokens.map((_, i) => i)
        .filter(i => !selectedTokenIndices.includes(i));
    
    if (unselectedIndices.length === 0) {
        showAlert('Please leave some tokens unselected for comparison.');
        return;
    }
    
    // Calculate mean embedding for selected tokens
    const selectedEmbeddings = selectedTokenIndices.map(i => tokenEmbeddings[i]);
    const meanSelected = selectedEmbeddings[0].map((_, j) => 
        selectedEmbeddings.reduce((sum, emb) => sum + emb[j], 0) / selectedEmbeddings.length
    );
    
    // Calculate mean embedding for unselected tokens
    const unselectedEmbeddings = unselectedIndices.map(i => tokenEmbeddings[i]);
    const meanUnselected = unselectedEmbeddings[0].map((_, j) => 
        unselectedEmbeddings.reduce((sum, emb) => sum + emb[j], 0) / unselectedEmbeddings.length
    );
    
    // Compute vector as difference between means (selected - unselected)
    const v = meanSelected.map((val, j) => val - meanUnselected[j]);
    
    // Normalize the vector
    const magnitude = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
        for (let i = 0; i < v.length; i++) {
            v[i] = v[i] / magnitude;
        }
    }
    
    // Create name prompt for the vector
    showPrompt("Enter a name for this vector (or cancel to not save it):", "", (name) => {
        if (name) {
            // Save the vector
            savedVectors.push({ name, vector: v });
            localStorage.setItem('savedVectors', JSON.stringify(savedVectors));
            
            // Update UI and switch to the new vector
            renderSavedVectors();
            selectedMode = 'saved';
            selectedSavedVectorIndex = savedVectors.length - 1;
            updateSelectionUI();
        } else {
            // Create a temporary vector without saving it
            const tempVector = v;
            
            // Use the temporary vector for coloring
            compositeWeights = [0, 0, 0, 0, 0, 0]; // Reset weights
            selectedMode = 'custom';
            selectedComponent = -1;
            
            // Apply the temporary vector for visualization
            currentTempVector = tempVector;
        }
        
        colorTokens();
        
        // Clear token selection
        selectedTokenIndices = [];
    });
});

// Process input text on submit
submitButton.addEventListener("click", () => {
    const inputText = textInput.value.trim();
    if (!inputText) {
        outputElement.textContent = "Please enter some text";
        tokensDisplay.textContent = "";
        pcaContainer.classList.add('hidden');
        pcaGridsContainer.classList.add('hidden');
        return;
    }
    
    // Show processing indicator
    processingIndicator.classList.remove('hidden');
    outputElement.textContent = "";
    tokensDisplay.textContent = "";
    
    // Use setTimeout to ensure UI updates before processing starts
    setTimeout(async () => {
        try {
            // Tokenize input
            const tokenizeResult = await tokenizer(inputText, { return_tensors: 'pt' });
            const tokenIds = Array.from(tokenizeResult.input_ids.data);
            
            // Decode tokens
            currentTokens = [];
            for (let i = 0; i < tokenIds.length; i++) {
                const token = tokenizer.decode([tokenIds[i]], { skip_special_tokens: false });
                currentTokens.push(token);
            }
            
            // Get embeddings
            const embeddings = await model(inputText, { pooling: 'none' });
            
            // Extract token embeddings
            const [batch_size, sequence_length, embedding_size] = embeddings.dims;
            tokenEmbeddings = [];
            for (let i = 0; i < sequence_length; i++) {
                const start = i * embedding_size;
                const end = start + embedding_size;
                const embedding = Array.from(embeddings.data.slice(start, end));
                tokenEmbeddings.push(embedding);
            }
            
            // Remove [CLS] and [SEP] tokens (first and last tokens)
            if (currentTokens.length > 2) {
                if (currentTokens[0] === '[CLS]') {
                    currentTokens = currentTokens.slice(1);
                    tokenEmbeddings = tokenEmbeddings.slice(1);
                }
                
                if (currentTokens[currentTokens.length - 1] === '[SEP]') {
                    currentTokens = currentTokens.slice(0, -1);
                    tokenEmbeddings = tokenEmbeddings.slice(0, -1);
                }
            }
            
            // Check if we have enough tokens for analysis
            if (currentTokens.length <= 1) {
                processingIndicator.classList.add('hidden');
                showAlert("Please enter more than one token for analysis.");
                tokensDisplay.textContent = currentTokens.join(' ');
                pcaContainer.classList.add('hidden');
                pcaGridsContainer.classList.add('hidden');
                return;
            }
            
            // Calculate PCA with at least 6 components if possible
            const numComponents = Math.min(6, tokenEmbeddings.length - 1);
            currentPcaResult = calculatePCA(tokenEmbeddings, numComponents);
            
            // Initialize grid points only if they haven't been moved yet
            if (!gridPoints[0].hasBeenMoved && !gridPoints[1].hasBeenMoved && !gridPoints[2].hasBeenMoved) {
                compositeWeights = [1, 0, 0, 0, 0, 0];
                currentWeightsElement.textContent = compositeWeights.map(w => w.toFixed(2)).join(', ');
            }
            
            // Hide processing indicator
            processingIndicator.classList.add('hidden');
            
            // Render UI
            renderPcaSelectors(currentPcaResult);
            
            // If we're using custom vector (grid points have been moved), update with that
            if (selectedComponent === -1) {
                // Recalculate weights from grid points
                selectedMode = 'grids';
                updateWeightsFromGrids();
                colorTokens();
            } else {
                // Default to showing the first component
                selectedMode = 'pca';
                selectedComponent = 0;
                updateSelectionUI();
                colorTokens();
            }
            
            // Only show Find Vector button when tokens are selected
            document.getElementById('find-vector').classList.add('hidden');
            
            // Show saved vectors container if we have any saved vectors
            if (savedVectors.length > 0) {
                document.getElementById('saved-vectors-container').classList.remove('hidden');
            }
            
            // Reset token selection state when processing new text
            selectedTokenIndices = [];
            
            outputElement.textContent = `Processed ${currentTokens.length} tokens successfully.`;
        } catch (error) {
            console.error("Error processing input:", error);
            processingIndicator.classList.add('hidden');
            outputElement.textContent = "Error: " + error.message;
        }
    }, 10); // Small delay to let the UI update
});

// Custom popup functions
function showAlert(message, callback = null) {
    const popupContainer = document.getElementById('popup-container');
    const popupContent = document.getElementById('popup-content');
    const popupOk = document.getElementById('popup-ok');
    const popupCancel = document.getElementById('popup-cancel');
    const popupInput = document.getElementById('popup-input');
    
    // Set content and hide unnecessary elements
    popupContent.textContent = message;
    popupCancel.classList.add('hidden');
    popupInput.classList.add('hidden');
    
    // Show popup
    popupContainer.classList.remove('hidden');
    
    // Handle OK button
    popupOk.onclick = () => {
        popupContainer.classList.add('hidden');
        if (callback) callback();
    };
}

function showPrompt(message, defaultValue = '', callback) {
    const popupContainer = document.getElementById('popup-container');
    const popupContent = document.getElementById('popup-content');
    const popupOk = document.getElementById('popup-ok');
    const popupCancel = document.getElementById('popup-cancel');
    const popupInput = document.getElementById('popup-input');
    
    // Set content and show input and cancel button
    popupContent.textContent = message;
    popupInput.value = defaultValue;
    popupInput.classList.remove('hidden');
    popupCancel.classList.remove('hidden');
    
    // Show popup
    popupContainer.classList.remove('hidden');
    
    // Set focus on input
    setTimeout(() => popupInput.focus(), 50);
    
    // Handle buttons
    popupOk.onclick = () => {
        const value = popupInput.value.trim();
        popupContainer.classList.add('hidden');
        callback(value);
    };
    
    popupCancel.onclick = () => {
        popupContainer.classList.add('hidden');
        callback(null);
    };
    
    // Handle Enter key in input
    popupInput.onkeydown = (e) => {
        if (e.key === 'Enter') {
            const value = popupInput.value.trim();
            popupContainer.classList.add('hidden');
            callback(value);
        }
    };
}

// Initialize saved vectors list on page load
renderSavedVectors();
