import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';

function App() {
    const [query, setQuery] = useState('');
    const [embedding, setEmbedding] = useState('e5');
    const [answer, setAnswer] = useState('');
    const [history, setHistory] = useState([]);
    const [structuredData, setStructuredData] = useState({
        structured_locations: [],
        structured_time_periods: [],
        structured_rulers_or_polities: []
    });
    const [retrievedDocs, setRetrievedDocs] = useState([]);
    const [expandedDocs, setExpandedDocs] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showStructured, setShowStructured] = useState(true);
    const [showDocuments, setShowDocuments] = useState(true);
    const useMock = false; // set to false to use real backend


    const handleSubmit = async () => {
        if (!query.trim()) return;
        setLoading(true);
        setError(null);

        try {
            let data;

            if (useMock) {
                data = {
                    answer: "Mocked answer to the question: " + query,
                    structured_locations: [],
                    structured_time_periods: [],
                    structured_rulers_or_polities: [],
                    full_document_contents: ["Mock document 1 content.", "Another sample document content."],
                    retrieved_documents: [
                        { id: "doc1", score: 0.98 },
                        { id: "doc2", score: 0.87 }
                    ]
                };
            } else {
                const res = await fetch('/api/rag-query/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, embedding })
                });

                data = await res.json();
                if (!res.ok) throw new Error(data.error || 'API Error');
            }

            const newPair = {
                question: query,
                answer: data.answer,
                documents: data.full_document_contents.map((content, i) => ({
                    content,
                    meta: data.retrieved_documents[i]
                }))
            };

            setHistory((prev) => [...prev, newPair]);

            setAnswer(data.answer);
            setStructuredData({
                structured_locations: data.structured_locations,
                structured_time_periods: data.structured_time_periods,
                structured_rulers_or_polities: data.structured_rulers_or_polities
            });
            setRetrievedDocs(newPair.documents);
        } catch (err) {
            setError(err.message || 'Network error');
        } finally {
            setLoading(false);
            setQuery('');
        }
    };

    const handleClear = () => {
        setQuery('');
        setAnswer('');
        setHistory([]);
        setStructuredData({
            structured_locations: [],
            structured_time_periods: [],
            structured_rulers_or_polities: []
        });
        setRetrievedDocs([]);
        setExpandedDocs({});
        setError(null);
    };

    const handleNewChat = async () => {
        await fetch('/api/clear-chat/', { method: 'POST' });
        handleClear();
    };

    const toggleExpand = (index) => {
        setExpandedDocs((prev) => ({ ...prev, [index]: !prev[index] }));
    };

    const hasStructuredInfo =
        structuredData.structured_locations.length > 0 ||
        structuredData.structured_time_periods.length > 0 ||
        structuredData.structured_rulers_or_polities.length > 0;

    return (
        <div className="p-4 max-w-4xl mx-auto space-y-6">
            <h1 className="text-2xl font-bold">RAG Historical Search</h1>

            <div className="space-y-3">
                {history.map((item, idx) => (
                    <div key={idx} className="bg-white border rounded p-4 shadow">
                        <p className="font-semibold text-blue-600">Q: {item.question}</p>
                        <p className="text-gray-800 mt-1">A: {item.answer}</p>
                        {hasStructuredInfo && (
                            <div className="bg-white p-4 border rounded shadow">
                                <div className="flex justify-between items-center">
                                    <h2 className="font-semibold text-lg">Structured Information</h2>
                                    <button
                                        onClick={() => setShowStructured(!showStructured)}
                                        className="text-sm text-blue-500 hover:underline"
                                    >
                                        {showStructured ? 'Hide' : 'Show'}
                                    </button>
                                </div>
                                {showStructured && (
                                    <div className="mt-2 space-y-2">
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
                                )}
                            </div>
                        )}

                        {retrievedDocs.length > 0 && (
                            <div className="bg-white p-4 border rounded shadow">
                                <div className="flex justify-between items-center">
                                    <h2 className="font-semibold text-lg">Retrieved Documents</h2>
                                    <button
                                        onClick={() => setShowDocuments(!showDocuments)}
                                        className="text-sm text-blue-500 hover:underline"
                                    >
                                        {showDocuments ? 'Hide' : 'Show'}
                                    </button>
                                </div>
                                {showDocuments && retrievedDocs.map((doc, i) => (
                                    <div key={i} className="mb-4 p-3 border rounded bg-gray-50">
                                        <div className="flex justify-between text-sm text-gray-600">
                                            <span>Doc {i + 1} | Score: {doc.meta.score.toFixed(2)}</span>
                                            <span>{doc.meta?.file_path?.split('/').pop() || 'Unknown Source'}</span>
                                        </div>
                                        <p className="mt-2 text-gray-800">
                                            {expandedDocs[i]
                                                ? doc.content
                                                : doc.content.slice(0, 300) + (doc.content.length > 300 ? '...' : '')
                                            }
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
                        )}
                    </div>
                ))}
            </div>

            <div className="flex flex-col md:flex-row md:space-x-2 space-y-2 md:space-y-0 pt-4">
                <input
                    type="text"
                    placeholder="Ask a question..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                    className="flex-1 p-2 border rounded"
                    disabled={loading}
                />
                <select
                    value={embedding}
                    onChange={(e) => setEmbedding(e.target.value)}
                    className="p-2 border rounded md:w-40"
                    disabled={loading}
                >
                    <option value="e5">E5</option>
                    <option value="mpnet">MPNet</option>
                </select>
                <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
                >
                    {loading ? 'Loading...' : 'Search'}
                </button>
                <button
                    onClick={handleClear}
                    className="bg-gray-400 text-white px-4 py-2 rounded"
                    disabled={loading}
                >
                    Clear
                </button>
                <button
                    onClick={handleNewChat}
                    className="bg-orange-500 text-white px-4 py-2 rounded"
                    disabled={loading}
                >
                    New Chat
                </button>
            </div>

            {error && <div className="text-red-600">{error}</div>}
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));
