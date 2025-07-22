// frontend_static/js/App.js

const { useState, useEffect } = React;

function App() {
    const [query, setQuery] = useState('');
    const [answer, setAnswer] = useState('Your answer will appear here...');
    const [retrievedDocs, setRetrievedDocs] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleQueryChange = (event) => {
        setQuery(event.target.value);
    };

    const handleSubmit = async () => {
        if (!query.trim()) {
            setError("Please enter a question.");
            return;
        }

        setLoading(true);
        setError(null);
        setAnswer('Thinking...');
        setRetrievedDocs([]);

        try {
            const response = await fetch('/api/rag-query/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // Add CSRF token if you enable it later (Django's default)
                    // 'X-CSRFToken': getCookie('csrftoken'),
                },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();

            if (response.ok) {
                setAnswer(data.answer);
                setRetrievedDocs(data.retrieved_documents);
                if (data.structured_locations?.length > 0) {
                    console.log("🗺 Structured Map Data:", data.structured_locations);
                }
            } else {
                setError(data.error || 'An unknown error occurred.');
                setAnswer('Error: ' + (data.error || 'Please check console.'));
                console.error("API Error:", data);
            }
        } catch (err) {
            setError('Failed to connect to the server. Is the Django server running?');
            setAnswer('Error: Failed to connect.');
            console.error("Fetch Error:", err);
        } finally {
            setLoading(false);
        }
    };

    // Optional: Function to get CSRF token (needed if you enable CSRF protection)
    // function getCookie(name) {
    //     let cookieValue = null;
    //     if (document.cookie && document.cookie !== '') {
    //         const cookies = document.cookie.split(';');
    //         for (let i = 0; i < cookies.length; i++) {
    //             const cookie = cookies[i].trim();
    //             if (cookie.startsWith(name + '=')) {
    //                 cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
    //                 break;
    //             }
    //         }
    //     }
    //     return cookieValue;
    // }

    return (
        <div className="flex flex-col space-y-6">
            <h1 className="text-3xl font-bold text-center text-gray-800">RAG Search Engine</h1>

            {/* Query Input */}
            <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
                <input
                    type="text"
                    value={query}
                    onChange={handleQueryChange}
                    placeholder="Ask a question about your documents..."
                    className="flex-grow p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    disabled={loading}
                />
                <button
                    onClick={handleSubmit}
                    className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-md shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
                    disabled={loading}
                >
                    {loading ? 'Searching...' : 'Search'}
                </button>
            </div>

            {/* Error Display */}
            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-md relative" role="alert">
                    <strong className="font-bold">Error!</strong>
                    <span className="block sm:inline"> {error}</span>
                </div>
            )}

            {/* Answer Display */}
            <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
                <h2 className="text-xl font-semibold text-gray-700 mb-2">Answer:</h2>
                <p className="text-gray-800">{answer}</p>
            </div>

            {/* Retrieved Documents Display */}
            <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
                <h2 className="text-xl font-semibold text-gray-700 mb-2">Structured Locations:</h2>
                {data.structured_locations?.length > 0 ? (
                    <ul className="list-disc list-inside space-y-2">
                    {data.structured_locations.map((loc, index) => (
                        <li key={index} className="text-gray-700">
                        <span className="font-medium">{loc.location_name}:</span> {loc.description}
                        </li>
                    ))}
                    </ul>
                ) : (
                    <p className="text-gray-600">No map-relevant locations found.</p>
                )}
            </div>
        </div>
    );
}

// Mount the React app
ReactDOM.render(<App />, document.getElementById('root'));
