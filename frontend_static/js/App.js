import React, { useState } from 'react';

function App() {
    const [query, setQuery] = useState('');
    const [embedding, setEmbedding] = useState('e5');
    const [answer, setAnswer] = useState('Your answer will appear here...');
    const [structuredData, setStructuredData] = useState({
        structured_locations: [],
        structured_time_periods: [],
        structured_rulers_or_polities: []
    });
    const [retrievedDocs, setRetrievedDocs] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expandedDocs, setExpandedDocs] = useState({});

    const handleSubmit = async () => {
        if (!query.trim()) return;
        setLoading(true);
        setError(null);

        try {
            const res = await fetch('/api/rag-query/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, embedding })
            });

            const data = await res.json();

            if (res.ok) {
                setAnswer(data.answer);
                setStructuredData({
                    structured_locations: data.structured_locations,
                    structured_time_periods: data.structured_time_periods,
                    structured_rulers_or_polities: data.structured_rulers_or_polities
                });
                setRetrievedDocs(data.full_document_contents.map((content, i) => ({
                    content,
                    meta: data.retrieved_documents[i]
                })));
            } else {
                setError(data.error || 'An error occurred');
            }
        } catch (err) {
            setError('Network error');
        } finally {
            setLoading(false);
        }
    };

    const toggleExpand = (index) => {
        setExpandedDocs((prev) => ({ ...prev, [index]: !prev[index] }));
    };

    return (
        <div className="p-4 max-w-4xl mx-auto space-y-6">
            <h1 className="text-2xl font-bold">RAG Historical Search</h1>
            <div className="flex flex-col md:flex-row md:space-x-2 space-y-2 md:space-y-0">
                <input
                    type="text"
                    placeholder="Ask a question..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="flex-1 p-2 border rounded"
                    disabled={loading}
                />
                <select
                    value={embedding}
                    onChange={(e) => setEmbedding(e.target.value)}
                    className="p-2 border rounded md:w-40"
                    disabled={loading}
                >
                    <option value="mpnet">MPNet</option>
                    <option value="e5">E5</option>
                </select>
                <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
                >
                    {loading ? 'Loading...' : 'Search'}
                </button>
            </div>

            {error && <div className="text-red-600">{error}</div>}

            <div className="bg-white p-4 border rounded shadow">
                <h2 className="font-semibold text-lg mb-2">Answer</h2>
                <p className="text-gray-800">{answer}</p>
            </div>

            {(structuredData.structured_locations.length > 0 ||
              structuredData.structured_time_periods.length > 0 ||
              structuredData.structured_rulers_or_polities.length > 0) && (
                <div className="bg-white p-4 border rounded shadow">
                    <h2 className="font-semibold text-lg mb-2">Structured Information</h2>
                    <div>
                        {structuredData.structured_locations.length > 0 && (
                            <>
                                <strong>Locations:</strong>
                                <ul className="list-disc ml-5">
                                    {structuredData.structured_locations.map((loc, i) => (
                                        <li key={i}>{loc.name} - {loc.description}</li>
                                    ))}
                                </ul>
                            </>
                        )}
                        {structuredData.structured_time_periods.length > 0 && (
                            <>
                                <strong>Time Periods:</strong>
                                <ul className="list-disc ml-5">
                                    {structuredData.structured_time_periods.map((t, i) => (
                                        <li key={i}>{t.name} - {t.description}</li>
                                    ))}
                                </ul>
                            </>
                        )}
                        {structuredData.structured_rulers_or_polities.length > 0 && (
                            <>
                                <strong>Rulers/Polities:</strong>
                                <ul className="list-disc ml-5">
                                    {structuredData.structured_rulers_or_polities.map((r, i) => (
                                        <li key={i}>{r.name} - {r.description}</li>
                                    ))}
                                </ul>
                            </>
                        )}
                    </div>
                </div>
            )}

            <div className="bg-white p-4 border rounded shadow">
                <h2 className="font-semibold text-lg mb-2">Retrieved Documents</h2>
                {retrievedDocs.map((doc, i) => (
                    <div key={i} className="mb-4 p-3 border rounded bg-gray-50">
                        <div className="flex justify-between text-sm text-gray-600">
                            <span>Doc {i + 1} | Score: {doc.meta.score.toFixed(2)}</span>
                            <span>{doc.meta?.file_path?.split('/').pop() || 'Unknown Source'}</span>
                        </div>
                        <p className="mt-2 text-gray-800">
                            {expandedDocs[i] ? doc.content : doc.content.slice(0, 300) + (doc.content.length > 300 ? '...' : '')}
                        </p>
                        <button
                            onClick={() => toggleExpand(i)}
                            className="text-blue-500 mt-1 text-sm hover:underline"
                        >
                            {expandedDocs[i] ? 'Show Less' : 'Show More'}
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default App;