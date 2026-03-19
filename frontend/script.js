// State management
const state = {
    moviesData: [], // { id, title } from movies_1.json
    selectedMovies: [],
    searchResults: [],
    recommendations: []
};

// API Configuration - Ready for Flask backend integration
const API_CONFIG = {
    baseURL: 'http://localhost:5000/api', // Update with your Flask API URL
    endpoints: {
        recommendations: '/movies/recommendations'
    }
};

// Mock data for initial recommendations (IDs from movies_1.json)
const MOCK_RECOMMENDATIONS = [
    { id: '79132', title: 'Inception', rating: 4 },
    { id: '130219', title: 'The Dark Knight', rating: 5 },
    { id: '47', title: 'Seven (a.k.a. Se7en)', rating: 7 },
    { id: '6', title: 'Heat', rating: 4 },
    { id: '103688', title: 'Conjuring, The', rating: 8 }
];

// DOM Elements
const elements = {
    movieSearch: document.getElementById('movieSearch'),
    clearBtn: document.getElementById('clearBtn'),
    searchBtn: document.getElementById('searchBtn'),
    searchResults: document.getElementById('searchResults'),
    selectedMoviesSection: document.getElementById('selectedMoviesSection'),
    selectedMoviesList: document.getElementById('selectedMoviesList'),
    recommendedMoviesSection: document.getElementById('recommendedMoviesSection'),
    recommendedMoviesList: document.getElementById('recommendedMoviesList'),
    getRecommendationsBtn: document.getElementById('getRecommendationsBtn')
};

// Normalize movie title: move "The" from end to beginning and remove any rating patterns
// Examples: 
//   "Godfather, The" → "The Godfather"
//   "Godfather, The (1972)" → "The Godfather (1972)"
//   "Dark Knight. The (2008)" → "The Dark Knight (2008)"
//   "Lord of the Rings: The Return of the King, The (2003)" → "The Lord of the Rings: The Return of the King (2003)"
//   "Movie Name 5.125/10" → "Movie Name" (removes rating)
function normalizeMovieTitle(title) {
    if (!title) return title;
    
    let trimmed = title.trim();
    
    // First, remove any rating patterns (e.g., "5.125/10", "4.5/10", etc.)
    // This handles cases where ratings might be concatenated into the title
    trimmed = trimmed.replace(/\s*\d+\.?\d*\s*\/\s*10\s*/g, '').trim();
    
    // Pattern: ", The" or ". The" optionally followed by year in parentheses at the end
    // This handles: "Movie Name, The (2003)" or "Movie Name. The" or "Movie Name, The"
    // Use a more robust pattern that captures everything before ", The" or ". The"
    const pattern = /^(.+?)[,.]\s*[Tt]he(\s*\([^)]+\))?\s*$/;
    const match = trimmed.match(pattern);
    
    if (match) {
        const movieName = match[1].trim();
        const yearPart = (match[2] || '').trim(); // Year in parentheses if present
        return `The ${movieName}${yearPart}`;
    }
    
    return trimmed;
}

// Initialize the app
async function init() {
    await loadMoviesData();
    loadInitialRecommendations();
    setupEventListeners();
}

// Load movies from movies_1.json
async function loadMoviesData() {
    try {
        const response = await fetch('movies_1.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const json = await response.json();
        state.moviesData = Object.entries(json).map(([id, title]) => ({ id, title }));
        console.log(`Loaded ${state.moviesData.length} movies from movies_1.json`);
    } catch (error) {
        console.error('Failed to load movies_1.json:', error);
        console.error('Make sure you are running a local server (e.g., python server.py or python -m http.server)');
        console.error('Opening index.html directly with file:// protocol will cause CORS errors.');
        state.moviesData = [];
        // Show user-friendly error message
        const errorMsg = document.createElement('div');
        errorMsg.style.cssText = 'background-color: #FF4444; color: white; padding: 16px; margin: 20px; border-radius: 8px; text-align: center;';
        errorMsg.innerHTML = '<strong>Error:</strong> Could not load movies_1.json. Please run a local server (see README.md or run: python server.py)';
        document.body.insertBefore(errorMsg, document.body.firstChild);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Search input events
    elements.movieSearch.addEventListener('input', handleSearchInput);
    elements.movieSearch.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });

    // Button events
    elements.clearBtn.addEventListener('click', clearSearch);
    elements.searchBtn.addEventListener('click', performSearch);
    elements.getRecommendationsBtn.addEventListener('click', getRecommendations);

    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
        const searchContainer = document.querySelector('.search-container');
        if (!searchContainer?.contains(e.target) && 
            !elements.searchResults?.contains(e.target)) {
            hideSearchResults();
        }
    });
}

// Handle search input
function handleSearchInput(e) {
    const query = e.target.value.trim();
    
    if (query.length === 0) {
        hideSearchResults();
        return;
    }

    // Debounce search (wait 300ms after user stops typing)
    clearTimeout(handleSearchInput.timeout);
    handleSearchInput.timeout = setTimeout(() => {
        performSearch();
    }, 300);
}

// Perform movie search from movies_1.json
function performSearch() {
    const query = elements.movieSearch.value.trim();
    
    if (query.length === 0) {
        hideSearchResults();
        return;
    }

    const q = query.toLowerCase();
    state.searchResults = state.moviesData
        .filter(m => m.title.toLowerCase().includes(q))
        .slice(0, 50); // Limit to 50 results
    
    displaySearchResults();
}

// Display search results
function displaySearchResults() {
    if (state.searchResults.length === 0) {
        hideSearchResults();
        return;
    }

    elements.searchResults.innerHTML = '';
    elements.searchResults.classList.add('active');

    state.searchResults.forEach(movie => {
        const isSelected = state.selectedMovies.some(m => m.id === movie.id);
        
        const resultItem = document.createElement('div');
        resultItem.className = `search-result-item ${isSelected ? 'selected' : ''}`;
        resultItem.innerHTML = `
            <span>${normalizeMovieTitle(movie.title)}</span>
        `;
        
        resultItem.addEventListener('click', () => {
            if (!isSelected) {
                selectMovie(movie);
            }
        });
        
        elements.searchResults.appendChild(resultItem);
    });
}

// Hide search results
function hideSearchResults() {
    elements.searchResults.classList.remove('active');
}

// Clear search
function clearSearch() {
    elements.movieSearch.value = '';
    hideSearchResults();
    elements.movieSearch.focus();
}

// Select a movie
function selectMovie(movie) {
    // Check if movie is already selected
    if (state.selectedMovies.some(m => m.id === movie.id)) {
        return;
    }

    // Add movie to selected list with default rating (movie.id from movies_1.json)
    const selectedMovie = {
        id: movie.id,
        title: movie.title,
        rating: 5 // Default rating
    };

    state.selectedMovies.push(selectedMovie);
    updateSelectedMoviesDisplay();
    hideSearchResults();
    elements.movieSearch.value = '';
}

// Remove a movie from selection
function removeMovie(movieId) {
    state.selectedMovies = state.selectedMovies.filter(m => m.id !== movieId);
    updateSelectedMoviesDisplay();
}

// Update movie rating
function updateMovieRating(movieId, rating) {
    const movie = state.selectedMovies.find(m => m.id === movieId);
    if (movie) {
        // Validate rating (1-10)
        const numRating = parseInt(rating);
        if (numRating >= 1 && numRating <= 10) {
            movie.rating = numRating;
        } else if (rating === '') {
            // Allow empty input while typing
            return;
        } else {
            // Reset to previous value if invalid
            const movieElement = document.querySelector(`[data-movie-id="${movieId}"] .rating-input`);
            if (movieElement) {
                movieElement.value = movie.rating;
            }
        }
    }
}

// Update selected movies display
function updateSelectedMoviesDisplay() {
    if (state.selectedMovies.length === 0) {
        elements.selectedMoviesSection.style.display = 'none';
        elements.getRecommendationsBtn.style.display = 'none';
        return;
    }

    elements.selectedMoviesSection.style.display = 'block';
    elements.getRecommendationsBtn.style.display = 'block';
    elements.selectedMoviesList.innerHTML = '';

    state.selectedMovies.forEach(movie => {
        const movieItem = document.createElement('div');
        movieItem.className = 'selected-movie-item';
        movieItem.setAttribute('data-movie-id', movie.id);
        
        movieItem.innerHTML = `
            <div class="movie-checkbox-container">
                <input 
                    type="checkbox" 
                    class="movie-checkbox" 
                    checked 
                    disabled
                >
                <span class="movie-title">${normalizeMovieTitle(movie.title)}</span>
            </div>
            <input 
                type="number" 
                class="rating-input" 
                min="1" 
                max="10" 
                value="${movie.rating}"
                placeholder="Rating"
            >
            <button class="remove-movie-btn" title="Remove movie">×</button>
        `;

        // Add event listeners
        const ratingInput = movieItem.querySelector('.rating-input');
        const removeBtn = movieItem.querySelector('.remove-movie-btn');

        ratingInput.addEventListener('input', (e) => {
            updateMovieRating(movie.id, e.target.value);
        });

        ratingInput.addEventListener('blur', (e) => {
            if (e.target.value === '' || parseInt(e.target.value) < 1) {
                e.target.value = movie.rating || 5;
            }
        });

        removeBtn.addEventListener('click', () => {
            removeMovie(movie.id);
        });

        elements.selectedMoviesList.appendChild(movieItem);
    });
}

// Get recommendations from API - sends movieIDs to backend
async function getRecommendations() {
    if (state.selectedMovies.length === 0) {
        alert('Please select at least one movie first.');
        return;
    }

    const movieIDs = state.selectedMovies.map(m => m.id);

    // Log movieIDs to console for verification
    console.log('movieIDs sent to backend:', movieIDs);

    // Show loading state
    elements.getRecommendationsBtn.textContent = 'Loading...';
    elements.getRecommendationsBtn.disabled = true;

    try {
        // Send movieIDs to Flask backend
        const response = await fetch(`${API_CONFIG.baseURL}${API_CONFIG.endpoints.recommendations}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ movieIDs })
        });

        if (response.ok) {
            const data = await response.json();
            state.recommendations = data.recommendations || [];
        } else {
            throw new Error('Failed to get recommendations');
        }

        displayRecommendations();
    } catch (error) {
        console.error('Recommendations error:', error);
        // Fallback to mock data when backend is not available
        state.recommendations = mockGetRecommendations(movieIDs);
        displayRecommendations();
    } finally {
        elements.getRecommendationsBtn.textContent = 'Get Recommendations';
        elements.getRecommendationsBtn.disabled = false;
    }
}

// Mock recommendations function (used when Flask API is not available)
function mockGetRecommendations(movieIDs) {
    const mockRecs = [
        { id: '79132', title: 'Inception', rating: 4 },
        { id: '130219', title: 'The Dark Knight', rating: 5 },
        { id: '47', title: 'Seven (a.k.a. Se7en)', rating: 7 },
        { id: '6', title: 'Heat', rating: 4 },
        { id: '103688', title: 'Conjuring, The', rating: 8 }
    ];
    const selectedIds = new Set(movieIDs);
    return mockRecs.filter(m => !selectedIds.has(m.id));
}

// Display recommendations (different layout from selections - list in bordered container)
function displayRecommendations() {
    if (state.recommendations.length === 0) {
        elements.recommendedMoviesList.innerHTML = '<p class="empty-message">No recommendations available.</p>';
        return;
    }

    elements.recommendedMoviesList.innerHTML = '';

    state.recommendations.forEach(movie => {
        const movieItem = document.createElement('div');
        movieItem.className = 'recommended-movie-item';
        
        movieItem.innerHTML = `
            <span class="recommended-movie-title">${normalizeMovieTitle(movie.title)}</span>
        `;

        elements.recommendedMoviesList.appendChild(movieItem);
    });
}

// Load initial recommendations
function loadInitialRecommendations() {
    state.recommendations = MOCK_RECOMMENDATIONS;
    displayRecommendations();
}

// Export functions for Flask API integration
// When Flask API is ready, you can use these functions:
window.CinemindAPI = {
    searchMovies: performSearch,
    getRecommendations: getRecommendations,
    getSelectedMovies: () => state.selectedMovies,
    setRecommendations: (recommendations) => {
        state.recommendations = recommendations;
        displayRecommendations();
    }
};

// Initialize app when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
